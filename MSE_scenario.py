"""
MSE-dependent scenario
@author: Xiwen Huang
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import glob

# Load data from the dataset directory
def load_data(path, postfix, choose=None):
    files = sorted(glob.glob(path + postfix))
    if isinstance(choose, int):
        print(f"building: {files[choose]}")
        return pd.read_csv(files[choose])
    elif isinstance(choose, str):
        file = glob.glob(path + choose + postfix)[0]
        return pd.read_csv(file)
    elif isinstance(choose, list):
        return [pd.read_csv(glob.glob(path + file + postfix)[0]) for file in choose]
    elif choose is None:
        return pd.read_csv(files[np.random.randint(0, len(files))])
    return [pd.read_csv(file) for file in files]

# Transform the data into supervised format
def series_to_supervised(data, n_in=1, n_out=1, rate_in=1, rate_out=1, sel_in=None, sel_out=None, dropnan=True):
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -rate_in):
        cols.append(df[sel_in].shift(i))
        names += [('%s(t-%d)' % (var, i)) for var in sel_in]
    for i in range(0, n_out, rate_out):
        cols.append(df[sel_out].shift(-i))
        names += [('%s(t+%d)' % (var, i)) for var in sel_out]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Construct train/validation/test datasets from the data
def construct_dataset(df):
    n = len(df)
    train_df, val_df, test_df = df[:int(0.7*n)], df[int(0.7*n):int(0.9*n)], df[int(0.9*n):]
    train_ds = series_to_supervised(train_df, n_in=24*7, n_out=24, rate_in=24, rate_out=24, sel_in=['value'], sel_out=['month', 'weekday', 'day', 'hour', 'airTemp', 'value'])
    val_ds = series_to_supervised(val_df, n_in=24*7, n_out=24, rate_in=24, rate_out=24, sel_in=['value'], sel_out=['month', 'weekday', 'day', 'hour', 'airTemp', 'value'])
    test_ds = series_to_supervised(test_df, n_in=24*7, n_out=24, rate_in=24, rate_out=24, sel_in=['value'], sel_out=['month', 'weekday', 'day', 'hour', 'airTemp', 'value'])
    return train_ds, val_ds, test_ds

def Algorithm(choose, seed=42):
    np.random.seed(seed)
    path_in = './Hog_buildings/'
    df_list = load_data(path=path_in, postfix='*.csv', choose=choose)
    
    if len(df_list) != 2:
        raise ValueError("The choose list must contain exactly two datasets: one for training/testing and one for validation.")
    
    df_a = df_list[0]
    df_b = df_list[1]

    
    train_ds, _, test_ds = construct_dataset(df_a)
    _, val_ds, _ = construct_dataset(df_b)

    # Convert datasets to NumPy arrays
    train_ds = train_ds.to_numpy()
    val_ds = val_ds.to_numpy()
    test_ds = test_ds.to_numpy()

    # Subsample the datasets
    label_sample_size = 100
    unlabel_sample_size = 500
    validate_sample_size = 100

    train_indices = np.random.choice(train_ds.shape[0], size=label_sample_size, replace=False)
    val_indices = np.random.choice(val_ds.shape[0], size=unlabel_sample_size, replace=False)
    test_indices = np.random.choice(test_ds.shape[0], size=validate_sample_size, replace=False)

    train_ds = train_ds[train_indices]
    val_ds = val_ds[val_indices]
    test_ds = test_ds[test_indices]

    # Extract features and labels
    x_labelled = np.delete(train_ds, np.s_[7:13], axis=1)
    y_labelled = train_ds[:, 12]

    x_unlabelled = np.delete(val_ds, np.s_[7:13], axis=1)
    y_unlabelled = val_ds[:, 12]

    x_validated = np.delete(test_ds, np.s_[7:13], axis=1)
    y_validated = test_ds[:, 12]

    
    return x_labelled, x_unlabelled, y_labelled, y_unlabelled, x_validated, y_validated


# Function for active learning variance computation
def calculate_variance_of_coefficients(x, y):
    lr = LinearRegression().fit(x, y)
    residuals = y - lr.predict(x)
    noise_estimate = np.std(residuals)
    xT_x_inv = np.linalg.inv(np.dot(x.T, x))
    coeff_variance = noise_estimate ** 2 * xT_x_inv
    return coeff_variance, noise_estimate

# MSE computation 
def fit_model_and_compute_mse(model, x_validated, y_validated):
    y_pred = model.predict(x_validated)
    return mean_squared_error(y_validated, y_pred)

# Bootstrap and ambiguity function
def bootstrap_and_ambiguity(x_train, y_train, x_unlabeled, n_bootstrap, sampling_seed):
    models, predictions = [], []
    n = x_train.shape[0]
    for i in range(n_bootstrap):
        model = LinearRegression()
        np.random.seed(i + sampling_seed)
        train_idxs = np.random.choice(range(n), size=n, replace=True)
        model.fit(x_train[train_idxs], y_train[train_idxs])
        models.append(model)
        predictions.append(model.predict(x_unlabeled))
    variances = np.var(predictions, axis=0)
    return models, variances



# Active learning strategy
def vb_active_learning(x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, 
                            phi, B, beta, eta_j, price_model, max_iterations=500):
    """
    Active learning process with support for buyer-centric (BC) or seller-centric (SC) pricing models.
    
    Parameters:
    - phi: Buyer’s willingness to pay (WTP).
    - B: Budget limit.
    - beta: Target cumulative MSE reduction.
    - eta_j: Constant WTS for seller-centric pricing.
    - price_model: Pricing model ('BC' for buyer-centric, 'SC' for seller-centric).
    - max_iterations: Maximum number of iterations for active learning.
    """
    cumulative_mse_reduction = 0
    cumulative_budget = 0
    iteration = 0
    data_bought_number = 0
    
    mse_list = []
    data_bought_number_list = []
    cumulative_budget_list = []
    price_list = []
    selected_sellers = []

    # Train initial model
    model = LinearRegression().fit(x_labelled, y_labelled)
    initial_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
    previous_mse = initial_mse
    mse_list.append(initial_mse)
    data_bought_number_list.append(data_bought_number)
    cumulative_budget_list.append(cumulative_budget)
    
    print(f"VBAL: Initial MSE = {initial_mse}, Budget Limit = {B}, Beta Threshold = {beta}")

    # Active learning loop
    while cumulative_budget < B and initial_mse - cumulative_mse_reduction > beta and iteration < max_iterations:
        iteration += 1
  # Check if x_unlabelled is empty
        if x_unlabelled.shape[0] == 0:
            print("No more unlabeled data points available. Exiting.")
            break
        # Estimate variance reduction for each unlabelled data point
        coeff_variance, noise_estimate = calculate_variance_of_coefficients(x_labelled, y_labelled)
        variances_new = np.array([
            noise_estimate * np.dot(np.dot(x_unlabelled[j], coeff_variance), x_unlabelled[j].T)
            for j in range(x_unlabelled.shape[0])
        ])
        
        # Select the data point with the highest variance contribution
        max_variance_index = np.argmax(variances_new)
        x_max_var_point = x_unlabelled[max_variance_index].reshape(1, -1)
        y_max_var_point = y_unlabelled[max_variance_index]

        # Add the selected data point to the labelled dataset
        x_temp_labelled = np.vstack([x_labelled, x_max_var_point])
        y_temp_labelled = np.append(y_labelled, y_max_var_point)

        # Retrain the model and compute new MSE
        model = LinearRegression().fit(x_temp_labelled, y_temp_labelled)
        new_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
        l_j = previous_mse - new_mse
        eta_j = eta_j_list[max_variance_index]

        # Compute the price of the data point
        p_j = phi * l_j if price_model == 'BC' else eta_j

        # Check purchase conditions
        if l_j > 0 and phi >= eta_j/l_j:
            # Purchase the data point
            cumulative_budget += p_j
            cumulative_mse_reduction += l_j
            previous_mse = new_mse

            x_labelled, y_labelled = x_temp_labelled, y_temp_labelled
            data_bought_number += 1

            print(f"Iteration {iteration}: Purchased data: {data_bought_number}, Selected seller: {max_variance_index}, Price = {p_j}, New MSE = {new_mse}, MSE Reduction = {l_j}, Cumulative Budget = {cumulative_budget}")
            
            mse_list.append(new_mse)
            data_bought_number_list.append(data_bought_number)
            cumulative_budget_list.append(cumulative_budget)
            price_list.append(p_j)
            selected_sellers.append(max_variance_index)
        else:
            # Skip the data point (no purchase)
            print(f"Iteration {iteration}: Data point skipped (WTP < price or no reduction), Current MSE = {previous_mse}, Price = {p_j}")
        
        # Remove the selected point from the unlabelled set
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

        # Terminate if the budget or target is reached
        if cumulative_budget >= B or new_mse <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            print(f"  Variance reduction: {improve}")
  
            break

    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers


def random_sampling_corrected_strategy(x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi, eta_j, B, beta, price_model):
    cumulative_mse_reduction = 0
    cumulative_budget = 0
    mse_list = []
    data_bought_number = 0
    data_bought_number_list = []
    cumulative_budget_list = []
    price_list = []
    iteration = 0
    selected_sellers = []
    model = LinearRegression().fit(x_labelled, y_labelled)

    initial_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
    mse_list.append(initial_mse)
    print(f"RSC: Initial MSE = {initial_mse}, Budget Limit = {B}, Beta Threshold = {beta}")

    data_bought_number_list.append(data_bought_number)
    cumulative_budget_list.append(cumulative_budget)
    previous_mse = initial_mse

    # Set seed for reproducibility
    np.random.seed(123)

    while cumulative_budget < B and initial_mse - cumulative_mse_reduction > beta:
        iteration += 1
        if x_unlabelled.shape[0] == 0:
            print("No more unlabeled data points.")
            break

        # Randomly pick one data point (without any comparison)
        random_idx = np.random.choice(range(x_unlabelled.shape[0]))
        x_random_point = x_unlabelled[random_idx].reshape(1, -1)
        y_random_point = y_unlabelled[random_idx]

        # Add it to the labelled set
        x_temp_labelled = np.vstack([x_labelled, x_random_point])
        y_temp_labelled = np.append(y_labelled, y_random_point)
        model = LinearRegression().fit(x_temp_labelled, y_temp_labelled)
        # Retrain the model with the new data
        new_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
        l_j = previous_mse - new_mse
        eta_j = eta_j_list[random_idx]
       
# Pricing model
        if price_model == 'BC':
            if l_j > 0: 
                p_j = phi * l_j
                if p_j <= eta_j:
                    print(f"Iteration {iteration}:  p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f}), Skip.")
                    x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
                    y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)
                    continue
            else:
                p_j = 0
        else:
            p_j = eta_j

        cumulative_budget += p_j
        cumulative_budget_list.append(cumulative_budget)
        data_bought_number += 1
        data_bought_number_list.append(data_bought_number)
        price_list.append(p_j)
        selected_sellers.append(random_idx)

        if l_j > 0:
            # Update labelled set and remove the selected point from the unlabelled set
            x_labelled = x_temp_labelled
            y_labelled = y_temp_labelled
            x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)

            # Update cumulative variables
            cumulative_mse_reduction += l_j
            previous_mse = new_mse
            mse_list.append(new_mse)

            # Log details and update lists
            print(f"Iteration {iteration}:Data bought number: {data_bought_number}, Selected seller: {random_idx}, Price: {p_j:.4f}, Cumulative Budget: {cumulative_budget:.4f}, Variance: {new_mse:.6f}")

        else:
            # record unchanged variance 
            mse_list.append(previous_mse)
            print(f"Iteration {iteration}: selected seller {random_idx}, Price: {p_j:.4f}, "
            f"Cumulative budget: {cumulative_budget:.4f}, Variance (unchanged): {previous_mse:.6f}")
            x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)
            continue


        # Exit if the budget is exceeded or if the variance reduction is small enough
        if cumulative_budget >= B or new_mse  <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            print(f"  Variance reduction: {improve}")
 
            break


    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers

# Bootstrap sampling strategy
def qbc_active_learning(x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi, B, beta, eta_j,price_model, n_bootstrap,  sampling_seed):
    _, variances = bootstrap_and_ambiguity(x_labelled, y_labelled, x_unlabelled, n_bootstrap, sampling_seed)
    cumulative_mse_reduction, cumulative_budget, iteration, data_bought_number = 0, 0, 0, 0
    mse_list, data_bought_number_list, cumulative_budget_list = [], [], []
    price_list = []
    selected_sellers = []
    model = LinearRegression().fit(x_labelled, y_labelled)
    initial_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
    mse_list.append(initial_mse)
    data_bought_number_list.append(data_bought_number)
    cumulative_budget_list.append(cumulative_budget)
    previous_mse = initial_mse
    print(f"QBCAL: Initial MSE = {initial_mse}, Budget Limit = {B}, Beta Threshold = {beta}")

    while cumulative_budget < B and initial_mse - cumulative_mse_reduction > beta:
        iteration += 1
        if x_unlabelled.shape[0] == 0:
            break

        _, variances = bootstrap_and_ambiguity(x_labelled, y_labelled, x_unlabelled, n_bootstrap, sampling_seed)
        max_variance_index = np.argmax(variances)
        x_max_var_point = x_unlabelled[max_variance_index].reshape(1, -1)
        y_max_var_point = y_unlabelled[max_variance_index]

        x_temp_labelled = np.vstack([x_labelled, x_max_var_point])
        y_temp_labelled = np.append(y_labelled, y_max_var_point)

        model = LinearRegression().fit(x_temp_labelled, y_temp_labelled)
        new_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
        l_j = previous_mse - new_mse
        eta_j = eta_j_list[max_variance_index]

        if l_j <= 0 or eta_j / l_j > phi:
            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
            continue

        x_labelled = x_temp_labelled
        y_labelled = y_temp_labelled
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
        cumulative_mse_reduction += l_j
        data_bought_number += 1

        p_j = phi * l_j if price_model =='BC' else eta_j
        cumulative_budget += p_j
        previous_mse = new_mse
        mse_list.append(new_mse)
        data_bought_number_list.append(data_bought_number)
        cumulative_budget_list.append(cumulative_budget)
        price_list.append(p_j)
        selected_sellers.append(max_variance_index)
        print(f"QBC - Iteration {iteration}:Purchased data: {data_bought_number}, Selected seller {max_variance_index},  New MSE = {new_mse}, MSE reduction = {l_j}, Price: {p_j:.4f}, Cumulative Budget = {cumulative_budget}")

        if cumulative_budget >= B or initial_mse - cumulative_mse_reduction <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            print(f"  Variance reduction: {improve}")
            break

    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers

def plot_mse_reduction_comparison(vb_data_bought, vb_mse_list, rsc_data_bought, rsc_mse_list, qbc_data_bought, qbc_mse_list, filename):
    plt.figure(figsize=(14, 12))
    plt.plot(vb_data_bought, vb_mse_list, marker='o', label='VBAL', color='#d62728', linewidth=3, markersize=10)
    plt.plot(qbc_data_bought, qbc_mse_list, marker='s', label='QBCAL', color='#2ca02c', linewidth=3, markersize=10)
    plt.plot(rsc_data_bought, rsc_mse_list, marker='x', linestyle='-.', label='RSC (baseline)', color='#1f77b4', linewidth=3, markersize=10)

    plt.xlabel('Data points purchased', fontsize=40)
    plt.ylabel('MSE', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.text(len(rsc_data_bought) * 0.1, beta , f'MSE threshold = {beta:.2f}', color='red', fontsize=35, weight='bold')
    plt.axhline(y=beta, color='red', linestyle='--')
    plt.legend(fontsize=30)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(filename)
    plt.show()
    
    


def plot_budget_utilization_comparison(vb_data_bought, vb_budget_list, rsc_data_bought, rsc_budget_list, qbc_data_bought, qbc_budget_list, filename):
    plt.figure(figsize=(14, 12))
    plt.plot(vb_data_bought, vb_budget_list, marker='o', label='VBAL', color='#d62728', linewidth=3, markersize=10)
    plt.plot(qbc_data_bought, qbc_budget_list, marker='s', label='QBCAL', color='#2ca02c', linewidth=3, markersize=10)
    plt.plot(rsc_data_bought, rsc_budget_list, marker = 'x',linestyle='-.',  label='RSC (baseline)', color = '#1f77b4', linewidth=3, markersize=10)

    plt.axhline(y=B, color='red', linestyle='--')
    plt.text(len(rsc_data_bought) * 0.1, B+30 , f'Budget Limit = {B}', color='red', fontsize=35, weight='bold')
    plt.xlabel('Data points purchased', fontsize=40)
    plt.ylabel('Cumulative budget (£)', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=30)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(filename)
    plt.show()
 

# Load data
data_seed = 123
choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
x_labelled, x_unlabelled, y_labelled, y_unlabelled, x_validated, y_validated = Algorithm(choose2, seed = data_seed)

model = LinearRegression().fit(x_labelled, y_labelled)
initial_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
 
# Example parameters
phi = 50
B = 1200
beta = initial_mse * 0.8
eta_j_list = np.full(x_unlabelled.shape[0], 30)
n_bootstrap = 10
sampling_seed = 42
# Call each strategy with given parameters to obtain the required variables for plotting
vb_BC_data_bought, vb_BC_mse_list, vb_BC_budget_list, vb_BC_price_list, vb_BC_sellers = vb_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated,
    phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='BC'
)

vb_SC_data_bought, vb_SC_mse_list, vb_SC_budget_list, vb_SC_price_list, vb_SC_sellers = vb_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated,
    phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='SC'
)

rsc_BC_data_bought, rsc_BC_mse_list, rsc_BC_budget_list, rsc_BC_price_list, rsc_BC_sellers = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='BC'
)

rsc_SC_data_bought, rsc_SC_mse_list, rsc_SC_budget_list, rsc_SC_price_list, rsc_SC_sellers  = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='SC'
)

qbc_BC_data_bought, qbc_BC_mse_list, qbc_BC_budget_list, qbc_BC_price_list, qbc_BC_sellers = qbc_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='BC', n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
)

qbc_SC_data_bought, qbc_SC_mse_list, qbc_SC_budget_list, qbc_SC_price_list, qbc_SC_sellers = qbc_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='SC', n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
)


# Plotting Variance Reduction Comparisons
plot_mse_reduction_comparison(
    vb_BC_data_bought, vb_BC_mse_list,  rsc_BC_data_bought, rsc_BC_mse_list, qbc_BC_data_bought, qbc_BC_mse_list,
    filename='mse_reduction_comparison(BC).pdf'
)

plot_mse_reduction_comparison(
    vb_SC_data_bought, vb_SC_mse_list,  rsc_SC_data_bought, rsc_SC_mse_list, qbc_SC_data_bought, qbc_SC_mse_list,
    filename='mse_reduction_comparison(SC).pdf'
)

# Plotting Budget Utilization Comparisons
plot_budget_utilization_comparison(
    vb_BC_data_bought, vb_BC_budget_list, rsc_BC_data_bought, rsc_BC_budget_list, qbc_BC_data_bought, qbc_BC_budget_list,
    filename='budget_utilization_comparison(BC).pdf'
)

plot_budget_utilization_comparison(
    vb_SC_data_bought, vb_SC_budget_list, rsc_SC_data_bought, rsc_SC_budget_list, qbc_SC_data_bought, qbc_SC_budget_list,
    filename='budget_utilization_comparison(SC).pdf'
)


def plot_percentage_variance_reduction_comparison(
    vb_price_list, vb_mse_list,
    rsc_price_list, rsc_mse_list, qbc_price_list, qbc_mse_list, filename
):
    """
    Plot Δv/Δc (mse reduction per unit cost) against data purchase number.

    Parameters:
    - vb_price_list: List of VBAL prices.
    - vb_mse_list: List of VBAL mse reductions.
    - rs_price_list: List of RS prices.
    - rs_mse_list: List of RS mse reductions.
    - rsc_price_list: List of RSC prices.
    - rsc_mse_list: List of RSC mse reductions.
    - qbc_price_list: List of QBCAL prices.
    - qbc_mse_list: List of QBCAL mse reductions.
    """
    plt.figure(figsize=(14, 12))

    # Prepare data
    strategies = [
        ("VBAL", vb_price_list, vb_mse_list),
        ("RSC", rsc_price_list, rsc_mse_list),
        ("QBCAL", qbc_price_list, qbc_mse_list),
    ]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']  # Colors per strategy
    markers = ['o', 'x', '^', 's']  # Markers per strategy

    for (strategy_name, price_list, mse_list), color, marker in zip(strategies, colors, markers):
        # Calculate Δv and efficiency (Δv/Δc)
        delta_v = [mse_list[i - 1] - mse_list[i] for i in range(1, len(mse_list))]
        delta_c = price_list[0:]  # Cost is taken directly from the price list (from index 1 onward)
        efficiency = [v / c if c != 0 else 0 for v, c in zip(delta_v, delta_c)]

        # Plot Δv/Δc vs data purchase number
        purchase_numbers = list(range(1, len(efficiency) + 1))
        plt.plot(purchase_numbers, efficiency, 
                 marker=marker, label=strategy_name, 
                 color=color, linewidth=3, markersize=10)

    # Labels and legends
    plt.xlabel('Data points purchased', fontsize=40)
    plt.ylabel('ΔMSE/Δc', fontsize=40, labelpad=15)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=30, loc='upper right')
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(filename)
    plt.show()

plot_percentage_variance_reduction_comparison(
    vb_BC_price_list, vb_BC_mse_list,
    rsc_BC_price_list, rsc_BC_mse_list,
    qbc_BC_price_list, qbc_BC_mse_list,
    filename = "2_BC_mse_c_vs_purchase_number.pdf"
)

plot_percentage_variance_reduction_comparison(
    vb_SC_price_list, vb_SC_mse_list,
    rsc_SC_price_list, rsc_SC_mse_list,
    qbc_SC_price_list, qbc_SC_mse_list,
    filename = "2_SC_mse_c_vs_purchase_number.pdf"
)

def plot_sorted_price_bar_chart_with_sc(
    bc_price_list, sc_price_list, bc_sellers, sc_sellers, strategy_name, filename, global_min_y, global_max_y
):
    
    
    # Determine the intersection of seller IDs
    common_sellers = set(bc_sellers).intersection(sc_sellers)
    if not common_sellers:
        raise ValueError("No common sellers found between bc_price_list and sc_price_list.")

    # Filter BC and SC lists based on common sellers
    bc_filtered = [(seller, price) for seller, price in zip(bc_sellers, bc_price_list) if seller in common_sellers]
    sc_filtered = [(seller, price) for seller, price in zip(sc_sellers, sc_price_list) if seller in common_sellers]

    # Align SC prices with BC prices using common seller order
    bc_filtered.sort(key=lambda x: x[1], reverse=True)  # Sort BC prices in descending order
    aligned_bc_prices = [price for _, price in bc_filtered]
    aligned_sc_prices = [dict(sc_filtered)[seller] for seller, _ in bc_filtered]

    # Create ranks for bar positions
    ranks = np.arange(1, len(aligned_bc_prices) + 1)

    plt.figure(figsize=(14, 12))

    # Plot bar charts
    bar_width = 0.4
    plt.bar(ranks - bar_width / 2, aligned_bc_prices, bar_width, alpha=0.6, label='BC (Buyer-centric)', color='blue', edgecolor='blue')
    plt.bar(ranks + bar_width / 2, aligned_sc_prices, bar_width, alpha=0.6, label='SC (Seller-centric)', color='red', edgecolor='red')

    # Add labels and legend
    xtick_interval = max(1, len(aligned_bc_prices) // 5)  # Show a tick for every 20% of total sellers
    plt.xticks(ticks=ranks[::xtick_interval], labels=ranks[::xtick_interval], fontsize=48)
    plt.xlabel("Seller number (ordered by BC)", fontsize=50)
    plt.ylabel("Revenue [£]", fontsize=50)
    plt.legend(fontsize=50)
    plt.yticks(fontsize=48)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set consistent y-axis limits
    plt.ylim(global_min_y, global_max_y)

    # Add title and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



# Calculate the global min and max y-axis values
all_prices = vb_BC_price_list + vb_SC_price_list + \
             qbc_BC_price_list + qbc_SC_price_list + \
             rsc_BC_price_list + rsc_SC_price_list
global_min_y = min(all_prices) * 0.9  # Add a 10% margin below the minimum
global_max_y = max(all_prices) * 1.1  # Add a 10% margin above the maximum


# VBAL
plot_sorted_price_bar_chart_with_sc(
    vb_BC_price_list, vb_SC_price_list, vb_BC_sellers, vb_SC_sellers,  "VBAL", filename="2_price_bar_chart_vbal_bc_sc.pdf", global_min_y=global_min_y, global_max_y=global_max_y
)


# QBCAL
plot_sorted_price_bar_chart_with_sc(
    qbc_BC_price_list, qbc_SC_price_list, qbc_BC_sellers, qbc_SC_sellers,  "QBCAL", filename="2_price_bar_chart_qbcal_bc_sc.pdf", global_min_y=global_min_y, global_max_y=global_max_y
)

# RSC
plot_sorted_price_bar_chart_with_sc(
    rsc_BC_price_list, rsc_SC_price_list, rsc_BC_sellers, rsc_SC_sellers,  "RSC", filename="2_price_bar_chart_rsc_bc_sc.pdf", global_min_y=global_min_y, global_max_y=global_max_y
)
