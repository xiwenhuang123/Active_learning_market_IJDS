"""
Predictive-ability-focused scenario (building energy case study).
Author: Xiwen Huang

This script reproduces the following results in the paper:
- Figure 10
- Figure 11
- Figures 14a–14f
- Tables 6 and 7 (Appendix)

It also generates additional diagnostic plots that are not shown in the paper.
"""


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import time


# ============================ Paths & timer ============================
start_time = time.time()

THIS_DIR = Path(__file__).resolve().parent
PLOT_DIR = THIS_DIR / "Plots" / "MSEScenario"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = THIS_DIR / "data" / "Hog_buildings"



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

def Algorithm(choose, data_seed):
    np.random.seed(data_seed)
    path_in = str(DATA_DIR) + "/" 
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



# VBAL (variance-based active learning strategy) 
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
    
    # print(f"VBAL: Initial MSE = {initial_mse}, Budget Limit = {B}, Beta Threshold = {beta}")

    # Active learning loop
    while cumulative_budget < B and initial_mse - cumulative_mse_reduction > beta and iteration < max_iterations:
        iteration += 1
  # Check if x_unlabelled is empty
        if x_unlabelled.shape[0] == 0:
            # print("No more unlabeled data points available. Exiting.")
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
        this_eta = eta_j[max_variance_index] 

        # Compute the price of the data point
        p_j = phi * l_j if price_model == 'BC' else this_eta

        # Check purchase conditions
        if l_j > 0 and phi >= this_eta/l_j:
            # Purchase the data point
            cumulative_budget += p_j
            cumulative_mse_reduction += l_j
            previous_mse = new_mse

            x_labelled, y_labelled = x_temp_labelled, y_temp_labelled
            data_bought_number += 1

            # print(f"vbal: Iteration {iteration}: Purchased data: {data_bought_number}, Selected seller: {max_variance_index}, Price = {p_j}, New MSE = {new_mse}, MSE Reduction = {l_j}, Cumulative Budget = {cumulative_budget}")
            
            mse_list.append(new_mse)
            data_bought_number_list.append(data_bought_number)
            cumulative_budget_list.append(cumulative_budget)
            price_list.append(p_j)
            selected_sellers.append(max_variance_index)
        # else:
        #     # Skip the data point (no purchase)
            # print(f"vbal: Iteration {iteration}: Data point skipped (WTP < price or no reduction), Current MSE = {previous_mse}, Price = {p_j}")
        
        # Remove the selected point from the unlabelled set
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

        # Terminate if the budget or target is reached
        if cumulative_budget >= B or new_mse <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            # print(f"  Variance reduction: {improve}")
  
            break

    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers

# Random sampling corrected strategy 
def random_sampling_corrected_strategy(x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi, eta_j, B, beta, price_model, rsc_seed):
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
    # print(f"RSC: Initial MSE = {initial_mse}, Budget Limit = {B}, Beta Threshold = {beta}")

    data_bought_number_list.append(data_bought_number)
    cumulative_budget_list.append(cumulative_budget)
    previous_mse = initial_mse

    if rsc_seed is not None:
            np.random.seed(rsc_seed)
    while cumulative_budget < B and initial_mse - cumulative_mse_reduction > beta:
        iteration += 1
        if x_unlabelled.shape[0] == 0:
            # print("No more unlabeled data points.")
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
        this_eta = eta_j[random_idx] 
       
# Pricing model
        if price_model == 'BC':
            if l_j > 0: 
                p_j = phi * l_j
                if p_j <= this_eta:
                    # print(f"Iteration {iteration}:  p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f}), Skip.")
                    x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
                    y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)
                    continue
            else:
                p_j = 0
        else:
            p_j = this_eta

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
            # print(f"rsc: Iteration {iteration}:Data bought number: {data_bought_number}, Selected seller: {random_idx}, Price: {p_j:.4f}, Cumulative Budget: {cumulative_budget:.4f}, Variance: {new_mse:.6f}")

        else:
            # record unchanged variance 
            mse_list.append(previous_mse)
            # print(f"Iteration {iteration}: selected seller {random_idx}, Price: {p_j:.4f}, "
            # f"Cumulative budget: {cumulative_budget:.4f}, Variance (unchanged): {previous_mse:.6f}")
            x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)
            continue


        # Exit if the budget is exceeded or if the variance reduction is small enough
        if cumulative_budget >= B or new_mse  <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            # print(f"  Variance reduction: {improve}")
 
            break


    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers

# QBCAL (Query-by-committee acite learning strategy)
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
    # print(f"QBCAL: Initial MSE = {initial_mse}, Budget Limit = {B}, Beta Threshold = {beta}")

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
        this_eta = eta_j[max_variance_index] 

        if l_j <= 0 or this_eta / l_j > phi:
            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
            continue

        x_labelled = x_temp_labelled
        y_labelled = y_temp_labelled
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
        cumulative_mse_reduction += l_j
        data_bought_number += 1

        p_j = phi * l_j if price_model =='BC' else this_eta
        cumulative_budget += p_j
        previous_mse = new_mse
        mse_list.append(new_mse)
        data_bought_number_list.append(data_bought_number)
        cumulative_budget_list.append(cumulative_budget)
        price_list.append(p_j)
        selected_sellers.append(max_variance_index)
        # print(f"QBC - Iteration {iteration}:Purchased data: {data_bought_number}, Selected seller {max_variance_index},  New MSE = {new_mse}, MSE reduction = {l_j}, Price: {p_j:.4f}, Cumulative Budget = {cumulative_budget}")

        if cumulative_budget >= B or initial_mse - cumulative_mse_reduction <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            # print(f"  Variance reduction: {improve}")
            break

    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers

# add arrows and ticks for figures
def _add_axis_arrows(ax, lw=2.5):
    """
    Draws arrowheads for x→ and y↑ using axes-fraction coordinates.
    Keeps bottom/left spines so ticks align with axes.
    Arrow length/position is visually consistent across plots.
    """
    # x-axis arrow (→), from about 90% width to 106% width along bottom
    ax.annotate(
        '',
        xy=(1.07, 0.0),         # arrow tip (slightly outside)
        xycoords='axes fraction',
        xytext=(1.0, 0.0),     # tail (also outside)
        arrowprops=dict(
            arrowstyle='-|>,head_width=0.6,head_length=1.0',
            lw=lw,
            color='black'
        ),
        clip_on=False
    )

    # Y-axis arrow ↑
    # Both xytext and xy are ABOVE the axis box (y > 1),
    # same x (0). Again: short stub just outside.
    ax.annotate(
        '',
        xy=(0.0, 1.07),         # arrow tip (slightly above)
        xycoords='axes fraction',
        xytext=(0.0, 1.0),     # tail (also above)
        arrowprops=dict(
            arrowstyle='-|>,head_width=0.6,head_length=1.0',
            lw=lw,
            color='black'
        ),
        clip_on=False
    )

def save_fig_consistent(fig, filename, width=14, height=12, dpi=300):
    """
    Save figure with consistent canvas size (no auto-tight cropping),
    always under Plots/MSEScenario.
    """
    fig.set_size_inches(width, height)
    out_path = PLOT_DIR / filename
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0.1
    )


# ======================== Produces a reference-only mse reduction plot (not included in the paper) ========================
def plot_mse_reduction_comparison(vb_data_bought, vb_mse_list, rsc_data_bought, rsc_mse_list, qbc_data_bought, qbc_mse_list, filename):
    fig, ax = plt.subplots(figsize=(14, 12))
    plt.plot(vb_data_bought, vb_mse_list,  marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    plt.plot(qbc_data_bought, qbc_mse_list, marker='s', label='QBCAL', color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')
    plt.plot(rsc_data_bought, rsc_mse_list, marker='^', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')

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
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()
    
def plot_budget_utilization_comparison(vb_data_bought, vb_budget_list, rsc_data_bought, rsc_budget_list, qbc_data_bought, qbc_budget_list, filename):
    fig, ax = plt.subplots(figsize=(14, 12))
    plt.plot(vb_data_bought, vb_budget_list, 
             marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    plt.plot(qbc_data_bought, qbc_budget_list, 
              marker='s', label='QBCAL',color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')
    plt.plot(rsc_data_bought, rsc_budget_list, 
             marker='^', label='RSC', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')


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
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()
 

# Load data
data_seed = 123
choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
x_labelled, x_unlabelled, y_labelled, y_unlabelled, x_validated, y_validated = Algorithm(choose2, data_seed = data_seed)

model = LinearRegression().fit(x_labelled, y_labelled)
initial_mse = fit_model_and_compute_mse(model, x_validated, y_validated)
 
# Example parameters
sampling_seed = 42
rsc_seed = 123
phi = 50
B = 1200
beta = initial_mse * 0.8
eta_j_list = np.full(x_unlabelled.shape[0], 30)
n_bootstrap = 10
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
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='BC', rsc_seed=rsc_seed
)

rsc_SC_data_bought, rsc_SC_mse_list, rsc_SC_budget_list, rsc_SC_price_list, rsc_SC_sellers  = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model='SC', rsc_seed=rsc_seed
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


# ======================== Generates Figure 10 in the paper ========================
def plot_percentage_mse_reduction_comparison(
    vb_price_list, vb_mse_list,
    rsc_price_list, rsc_mse_list,
    qbc_price_list, qbc_mse_list,
    filename
):
    """
    Plot cumulative average ΔMSE/Δc vs. purchase number.

    For each strategy:
        ΔMSE_k   = MSE_{k-1} - MSE_{k}
        Δc_k     = price_k
        eff_k    = ΔMSE_k / Δc_k
        y[k]     = (eff_1 + ... + eff_k)/k   (running avg to smooth the curve)
    """

    fig, ax = plt.subplots(figsize=(14, 12))

    # bundle strategy data
    strategies = [
        ("VBAL",  vb_price_list,  np.array(vb_mse_list)*1e3),
        ("QBCAL", qbc_price_list, np.array(qbc_mse_list)*1e3),
        ("RSC",   rsc_price_list, np.array(rsc_mse_list)*1e3),

    ]

    colors  = ['#d62728', '#2ca02c', '#1f77b4']
    markers = ['o',        's',        '^'      ]

    for (strategy_name, price_list, mse_trace), color, marker in zip(strategies, colors, markers):
        # safety: skip if trace too short
        if len(mse_trace) < 2:
            continue

        # per-step improvement in MSE
        delta_mse = [
            mse_trace[i - 1] - mse_trace[i]
            for i in range(1, len(mse_trace))
        ]

        # per-step cost
        delta_c = price_list[:len(delta_mse)]

        # instantaneous efficiency for each acquired point
        point_eff = [
            (dm / dc) if dc != 0 else 0.0
            for dm, dc in zip(delta_mse, delta_c)
        ]

        # cumulative running average efficiency
        avg_eff = []
        running_sum = 0.0
        for k, eff in enumerate(point_eff, start=1):
            running_sum += eff
            avg_eff.append(running_sum / k)

        purchase_numbers = list(range(1, len(avg_eff) + 1))

        plt.plot(
            purchase_numbers,
            avg_eff,
            marker=marker,
            label=strategy_name,
            color=color,
            linewidth=3,
            markersize=25,
            markeredgewidth=1.2, markerfacecolor=color, markeredgecolor='white')

    plt.xlabel('Data point purchased', fontsize=35)
    plt.ylabel(r'$\Delta \mathrm{MSE} / \Delta c$ [×$10^{-3}$ TWD$^2$/£]', fontsize=35)
    max_purchase = max(
        len(vb_mse_list),
        len(qbc_mse_list),
        len(rsc_mse_list),

    )
    step = 4
    xticks = list(range(1, max_purchase + 1, step))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_purchase + 0.5)


    plt.xticks(xticks, fontsize=30)
    ax = plt.gca()

    # ticks
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)

    # cleaner y tick formatting (no long decimals)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}".rstrip('0').rstrip('.')))

    # legend (move slightly outside so it doesn't sit on the curve)
    plt.legend(fontsize=30)

    # remove grid and top/right spines; thicken main axes like before
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    _add_axis_arrows(ax, lw=2.5)
    # leave padding so arrowheads aren't cropped
    plt.subplots_adjust(left=0.075, right=0.988, top=0.93, bottom=0.15)

    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)


    plt.show()




plot_percentage_mse_reduction_comparison(
    vb_BC_price_list, vb_BC_mse_list,
    rsc_BC_price_list, rsc_BC_mse_list,
    qbc_BC_price_list, qbc_BC_mse_list,
    filename = "2_BC_mse_c_vs_purchase_number.pdf"
)

plot_percentage_mse_reduction_comparison(
    vb_SC_price_list, vb_SC_mse_list,
    rsc_SC_price_list, rsc_SC_mse_list,
    qbc_SC_price_list, qbc_SC_mse_list,
    filename = "2_SC_mse_c_vs_purchase_number.pdf"
)


# ======================== Generates Figure 11 in the paper ========================
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

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot bar charts
    bar_width = 0.4
    ax.bar(
        ranks - bar_width / 2,
        aligned_bc_prices,
        bar_width,
        alpha=0.6,
        label='BC (Buyer-centric)',
        color='blue',
        edgecolor='blue'
    )
    ax.bar(
        ranks + bar_width / 2,
        aligned_sc_prices,
        bar_width,
        alpha=0.6,
        label='SC (Seller-centric)',
        color='red',
        edgecolor='red'
    )
       # X ticks
    num_sellers = len(ranks)

    if num_sellers > 1:
        step = 4  # show every 4th seller
        xticks = list(range(1, num_sellers + 1, step))
        # include last tick only if it aligns nicely
        if (num_sellers - xticks[-1]) >= step / 2:
            xticks.append(num_sellers)
    else:
        xticks = [1]

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=48)

    # Labels
    ax.set_xlabel("Seller rank (ordered by BC)", fontsize=50)
    ax.set_ylabel("Revenue [£]", fontsize=50)

    # Ticks
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=48)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=48)

    # Y limits
    ax.set_ylim(global_min_y, global_max_y)

    # Frame
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    # Legend
    ax.legend(
        fontsize=50,
        loc='upper right'
    )

    # Axis arrows
    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.075, right=0.988, top=0.93, bottom=0.15)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()

# Calculate the global min and max y-axis values
all_prices = vb_BC_price_list + vb_SC_price_list + \
             qbc_BC_price_list + qbc_SC_price_list + \
             rsc_BC_price_list + rsc_SC_price_list
global_min_y = min(all_prices) * 0.9  # Add a 10% margin below the minimum
global_max_y = max(all_prices) * 1.1  # Add a 10% margin above the maximum

# Generate Figure 11a-VBAL
plot_sorted_price_bar_chart_with_sc(
    vb_BC_price_list, vb_SC_price_list, vb_BC_sellers, vb_SC_sellers,  "VBAL", filename="2_price_bar_chart_vbal_bc_sc.pdf", global_min_y=global_min_y, global_max_y=global_max_y
)

# # Generate Figure 11a-QBCAL
plot_sorted_price_bar_chart_with_sc(
    qbc_BC_price_list, qbc_SC_price_list, qbc_BC_sellers, qbc_SC_sellers,  "QBCAL", filename="2_price_bar_chart_qbcal_bc_sc.pdf", global_min_y=global_min_y, global_max_y=global_max_y
)

# # Generate Figure 11a-RSC
plot_sorted_price_bar_chart_with_sc(
    rsc_BC_price_list, rsc_SC_price_list, rsc_BC_sellers, rsc_SC_sellers,  "RSC", filename="2_price_bar_chart_rsc_bc_sc.pdf", global_min_y=global_min_y, global_max_y=global_max_y
)


# ======================== Generates Figure 14a-14f in the paper ========================

# =====Generates Figures 14a and 14b in the paper ======
def purchases_vs_phi_mse(phi_list, B_fixed, price_model="BC", seed=42):
    vbal_counts, qbcal_counts, rsc_counts = [], [], []
    choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
    x_lab, x_unlab, y_lab, y_unlab, x_val, y_val = Algorithm(choose2,data_seed=data_seed)

    model = LinearRegression().fit(x_lab, y_lab)
    init_mse = fit_model_and_compute_mse(model, x_val, y_val)
    beta_local = init_mse * 0.8

    eta_local = np.full(x_unlab.shape[0], 30.0)
    for i, phi_val in enumerate(phi_list):
        vb_db, _, _, _, _ = vb_active_learning(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi=phi_val, B=B_fixed, beta=beta_local,
            eta_j=eta_local.copy(), price_model=price_model
        )

        # QBCAL
        q_db, _, _, _, _ = qbc_active_learning(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi=phi_val, B=B_fixed, beta=beta_local,
            eta_j=eta_local.copy(), price_model=price_model,
            n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
        )

        # RSC
        r_db, _, _, _, _ = random_sampling_corrected_strategy(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi=phi_val, eta_j=eta_local.copy(),
            B=B_fixed, beta=beta_local,
            price_model=price_model,
            rsc_seed=rsc_seed+i
        )

        vbal_counts.append(vb_db[-1] if len(vb_db) else 0)
        qbcal_counts.append(q_db[-1] if len(q_db) else 0)
        rsc_counts.append(r_db[-1] if len(r_db) else 0)

    return np.array(vbal_counts), np.array(qbcal_counts), np.array(rsc_counts)


def plot_sensitivity_to_phi_mse(phi_list, B_fixed, price_model="BC", filename="sensitivity_to_phi.pdf"):
    vbal_counts, qbcal_counts, rsc_counts = purchases_vs_phi_mse(
        phi_list, B_fixed, price_model=price_model
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(phi_list, vbal_counts, marker='o', label='VBAL', color='#d62728',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#d62728', markeredgecolor='white')
    ax.plot(phi_list, qbcal_counts, marker='s', label='QBCAL', color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#2ca02c', markeredgecolor='white')
    ax.plot(phi_list, rsc_counts, marker='^', label='RSC', color='#1f77b4',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#1f77b4', markeredgecolor='white')

    ax.set_xlabel(r'Willingness to pay $\phi$ [£/TWD$^2$]', fontsize=35)
    ax.set_ylabel('Data points purchased', fontsize=35)
    ax.set_xticks(phi_list)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(fontsize=30, loc='best')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()

phi_values = [30, 40, 50, 60, 70]
plot_sensitivity_to_phi_mse(phi_values, B_fixed=1200, price_model="BC",
                            filename="sens_phi_BC.pdf")
plot_sensitivity_to_phi_mse(phi_values, B_fixed=1200, price_model="SC",
                            filename="sens_phi_SC.pdf")


# =====Generates Figures 14c and 14d in the paper ======
def purchases_vs_wts_scale_mse(scale_list, phi_fixed, B_fixed,
                               price_model="BC"):
    vbal_counts, qbcal_counts, rsc_counts = [], [], []

    # same data split for all scales
    choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
    x_lab, x_unlab, y_lab, y_unlab, x_val, y_val = Algorithm(choose2, data_seed=data_seed)

    model = LinearRegression().fit(x_lab, y_lab)
    init_mse = fit_model_and_compute_mse(model, x_val, y_val)
    beta_local = init_mse * 0.8

    for i, s in enumerate(scale_list):
        eta_local = np.full(x_unlab.shape[0], 30.0 * s)

        vb_db, _, _, _, _ = vb_active_learning(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi_fixed, B_fixed, beta_local, eta_local.copy(),
            price_model=price_model
        )

        q_db, _, _, _, _ = qbc_active_learning(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi_fixed, B_fixed, beta_local, eta_local.copy(),
            price_model=price_model,
            n_bootstrap=n_bootstrap,
            sampling_seed=sampling_seed
        )

        r_db, _, _, _, _ = random_sampling_corrected_strategy(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi_fixed, eta_local.copy(), B_fixed, beta_local,
            price_model=price_model,
            rsc_seed = rsc_seed+i
        )

        vbal_counts.append(vb_db[-1] if len(vb_db) else 0)
        qbcal_counts.append(q_db[-1] if len(q_db) else 0)
        rsc_counts.append(r_db[-1] if len(r_db) else 0)

    return np.array(vbal_counts), np.array(qbcal_counts), np.array(rsc_counts)


def plot_sensitivity_to_wts_scale_mse(scale_list, phi_fixed, B_fixed,
                                      price_model="BC",
                                      filename="sensitivity_to_wts.pdf"):
    vbal_counts, qbcal_counts, rsc_counts = purchases_vs_wts_scale_mse(
        scale_list, phi_fixed, B_fixed, price_model=price_model
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(scale_list, vbal_counts, marker='o', label='VBAL', color='#d62728',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#d62728', markeredgecolor='white')
    ax.plot(scale_list, qbcal_counts, marker='s', label='QBCAL', color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#2ca02c', markeredgecolor='white')
    ax.plot(scale_list, rsc_counts, marker='^', label='RSC', color='#1f77b4',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#1f77b4', markeredgecolor='white')

    ax.set_xlabel(r'WTS scaling factor', fontsize=35)
    ax.set_ylabel('Data points purchased', fontsize=35)
    ax.set_xticks(scale_list)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(fontsize=30, loc='lower left')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()
wts_scales = [1, 2, 3, 4, 5]
plot_sensitivity_to_wts_scale_mse(
    wts_scales, phi_fixed=50, B_fixed=1200,
    price_model="BC",
    filename="sens_wts_BC.pdf"
)
plot_sensitivity_to_wts_scale_mse(
    wts_scales, phi_fixed=50, B_fixed=1200,
    price_model="SC",
    filename="sens_wts_SC.pdf"
)


# =====Generates Figures 14e and 14f in the paper ======
def purchases_vs_budget_mse(B_list, price_model="BC", phi=50):
    vbal_counts, qbcal_counts, rsc_counts = [], [], []
    # fixed split
    choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
    x_lab, x_unlab, y_lab, y_unlab, x_val, y_val = Algorithm(choose2, data_seed=data_seed)

    model = LinearRegression().fit(x_lab, y_lab)
    init_mse = fit_model_and_compute_mse(model, x_val, y_val)

    # MAIN EXPERIMENT USES 0.8, BUT HERE WE WANT TO FORCE BUDGET TO BIND:
    beta_local = init_mse * 0.1  

    for i, B_val in enumerate(B_list):
        eta_local = np.full(x_unlab.shape[0], 30.0)

        vb_db, _, _, _, _ = vb_active_learning(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi, B_val, beta_local, eta_local.copy(),
            price_model=price_model
        )

        q_db, _, _, _, _ = qbc_active_learning(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi, B_val, beta_local, eta_local.copy(),
            price_model=price_model,
            n_bootstrap=n_bootstrap,
            sampling_seed=sampling_seed
        )

        r_db, _, _, _, _ = random_sampling_corrected_strategy(
            x_lab.copy(), y_lab.copy(),
            x_unlab.copy(), y_unlab.copy(),
            x_val, y_val,
            phi, eta_local.copy(), B_val, beta_local,
            price_model=price_model,
            rsc_seed = rsc_seed+i
        )

        vbal_counts.append(vb_db[-1] if len(vb_db) else 0)
        qbcal_counts.append(q_db[-1] if len(q_db) else 0)
        rsc_counts.append(r_db[-1] if len(r_db) else 0)

    return np.array(vbal_counts), np.array(qbcal_counts), np.array(rsc_counts)

def plot_sensitivity_to_budget_mse(B_list, price_model="BC", filename="sensitivity_to_budget.pdf"):
    vbal_counts, qbcal_counts, rsc_counts = purchases_vs_budget_mse(
        B_list, price_model=price_model
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(B_list, vbal_counts, marker='o', label='VBAL', color='#d62728',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#d62728', markeredgecolor='white')
    ax.plot(B_list, qbcal_counts, marker='s', label='QBCAL', color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#2ca02c', markeredgecolor='white')
    ax.plot(B_list, rsc_counts, marker='^', label='RSC', color='#1f77b4',
            linewidth=3, markersize=25, markeredgewidth=1.2,
            markerfacecolor='#1f77b4', markeredgecolor='white')

    ax.set_xlabel(r'Budget limit $B$ [£]', fontsize=35)
    ax.set_ylabel('Data points purchased', fontsize=35)
    ax.set_xticks(B_list)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(fontsize=30, loc='upper left')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()
B_values = [800, 1000, 1200, 1400, 1600]
plot_sensitivity_to_budget_mse(B_values, price_model="BC", filename="sens_B_BC.pdf")
plot_sensitivity_to_budget_mse(B_values, price_model="SC", filename="sens_B_SC.pdf")


# ============================================================
# Appendix: Generate Tables 6 and 7 (Monte Carlo robustness evaluation, 50 independent partitions)
# ============================================================

def _run_one_al_split(seed, price_model, phi, B, eta_scale=1.0):
    data_seed = seed
    choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
    x_lab, x_unlab, y_lab, y_unlab, x_val, y_val = Algorithm(choose2, data_seed=data_seed)

    model = LinearRegression().fit(x_lab, y_lab)
    init_mse = fit_model_and_compute_mse(model, x_val, y_val)
    beta_local = init_mse * 0.8

    eta_local = np.full(x_unlab.shape[0], 30.0 * eta_scale)

    vb_db, _, _, _, _ = vb_active_learning(
        x_lab.copy(), y_lab.copy(),
        x_unlab.copy(), y_unlab.copy(),
        x_val, y_val,
        phi=phi, B=B, beta=beta_local, eta_j=eta_local.copy(),
        price_model=price_model
    )

    bootstrap_seed = seed + 1000
    q_db, _, _, _, _ = qbc_active_learning(
        x_lab.copy(), y_lab.copy(),
        x_unlab.copy(), y_unlab.copy(),
        x_val, y_val,
        phi=phi, B=B, beta=beta_local, eta_j=eta_local.copy(),
        price_model=price_model,
        n_bootstrap=n_bootstrap,
        sampling_seed=bootstrap_seed
    )

    rsc_seed = seed + 2000
    r_db, _, _, _, _ = random_sampling_corrected_strategy(
        x_lab.copy(), y_lab.copy(),
        x_unlab.copy(), y_unlab.copy(),
        x_val, y_val,
        phi=phi, eta_j=eta_local.copy(), B=B, beta=beta_local,
        price_model=price_model, rsc_seed=rsc_seed
    )

    vb_last = vb_db[-1] if vb_db else 0
    q_last  = q_db[-1] if q_db else 0
    r_last  = r_db[-1] if r_db else 0
    return vb_last, q_last, r_last

def monte_carlo_summary(param_list, B_fixed, price_model="BC", runs=50, param_name="phi"):
    records = []
    for val in param_list:
        vb_counts, q_counts, r_counts = [], [], []
        for s in range(runs):
            vb_c, q_c, r_c = _run_one_al_split(
                seed=100 + s,
                price_model=price_model,
                phi=val if param_name == "phi" else 50,
                B=val if param_name == "B" else B_fixed,
                eta_scale=val if param_name == "eta" else 1.0
            )
            vb_counts.append(vb_c)
            q_counts.append(q_c)
            r_counts.append(r_c)

        def summarize(arr):
            arr = np.array(arr)
            return np.mean(arr), np.percentile(arr, 25), np.percentile(arr, 75)

        vb_mean, vb_p25, vb_p75 = summarize(vb_counts)
        qb_mean, qb_p25, qb_p75 = summarize(q_counts)
        r_mean,  r_p25,  r_p75  = summarize(r_counts)

        records.append({
            param_name: val,
            "VBAL_mean": vb_mean, "VBAL_p25": vb_p25, "VBAL_p75": vb_p75,
            "QBCAL_mean": qb_mean, "QBCAL_p25": qb_p25, "QBCAL_p75": qb_p75,
            "RSC_mean":  r_mean,  "RSC_p25":  r_p25,  "RSC_p75":  r_p75
        })
    return pd.DataFrame(records)
phi_values = [30, 40, 50, 60, 70]
wts_scales = [1, 2, 3, 4, 5]
B_values = [800, 1000, 1200, 1400, 1600]
B_fixed = 1200

# WTP (φ)
df_wtp_bc = monte_carlo_summary(phi_values, B_fixed, price_model="BC", runs=50, param_name="phi")
df_wtp_sc = monte_carlo_summary(phi_values, B_fixed, price_model="SC", runs=50, param_name="phi")

# WTS scaling
df_wts_bc = monte_carlo_summary(wts_scales, B_fixed, price_model="BC", runs=50, param_name="eta")
df_wts_sc = monte_carlo_summary(wts_scales, B_fixed, price_model="SC", runs=50, param_name="eta")

# Budget
df_b_bc = monte_carlo_summary(B_values, B_fixed=None, price_model="BC", runs=50, param_name="B")
df_b_sc = monte_carlo_summary(B_values, B_fixed=None, price_model="SC", runs=50, param_name="B")

print("\n=== Sensitivity on WTP (φ) — Buyer-Centric ===")
print(df_wtp_bc.round(2))

print("\n=== Sensitivity on WTP (φ) — Seller-Centric ===")
print(df_wtp_sc.round(2))

print("\n=== Sensitivity on WTS scaling (η) — Buyer-Centric ===")
print(df_wts_bc.round(2))

print("\n=== Sensitivity on WTS scaling (η) — Seller-Centric ===")
print(df_wts_sc.round(2))

print("\n=== Sensitivity on Budget (B) — Buyer-Centric ===")
print(df_b_bc.round(2))

print("\n=== Sensitivity on Budget (B) — Seller-Centric ===")
print(df_b_sc.round(2))

RESULT_DIR = THIS_DIR / "Plots" / "Sensitivity_Results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

df_wtp_bc.to_csv(RESULT_DIR / "table6_wtp_BC.csv", index=False)
df_wtp_sc.to_csv(RESULT_DIR / "table6_wtp_SC.csv", index=False)
df_wts_bc.to_csv(RESULT_DIR / "table7_wts_BC.csv", index=False)
df_wts_sc.to_csv(RESULT_DIR / "table7_wts_SC.csv", index=False)
df_b_bc.to_csv(RESULT_DIR / "table7_budget_BC.csv", index=False)
df_b_sc.to_csv(RESULT_DIR / "table7_budget_SC.csv", index=False)

# ============================ Total runtime ============================
end_time = time.time()
print("\n===================================")
print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")
print("===================================\n")


# # ===============================
# # Wilcoxon tests on Avg. Cost (MSE-dependent scenario)
# # ===============================
# from scipy.stats import wilcoxon

# def avg_cost(prices):
#     return float(np.mean(prices)) if len(prices) > 0 else np.nan

# def run_one_mse_split(seed, price_model):
#     # Same two buildings; variability comes from subsampling with the seed
#     choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
#     x_lab, x_unlab, y_lab, y_unlab, x_val, y_val = Algorithm(choose2, seed=seed)

#     # Per-run threshold (your definition): beta = 0.8 * initial MSE
#     model = LinearRegression().fit(x_lab, y_lab)
#     init_mse = fit_model_and_compute_mse(model, x_val, y_val)
#     beta_local = init_mse * 0.8

#     # Per-run eta_j (constant 30 as in your script)
#     eta_run = np.full(x_unlab.shape[0], 30.0)

#     # Run the three strategies on the SAME split (pairing unit = run)
#     vb_db, vb_mse, vb_budget, vb_prices, _ = vb_active_learning(
#         x_lab.copy(), y_lab.copy(), x_unlab.copy(), y_unlab.copy(), x_val, y_val,
#         phi=50, B=1200, beta=beta_local, eta_j=eta_run, price_model=price_model
#     )

#     q_db, q_mse, q_budget, q_prices, _ = qbc_active_learning(
#         x_lab.copy(), y_lab.copy(), x_unlab.copy(), y_unlab.copy(), x_val, y_val,
#         phi=50, B=1200, beta=beta_local, eta_j=eta_run, price_model=price_model,
#         n_bootstrap=10, sampling_seed=42
#     )

#     r_db, r_mse, r_budget, r_prices, _ = random_sampling_corrected_strategy(
#         x_lab.copy(), y_lab.copy(), x_unlab.copy(), y_unlab.copy(), x_val, y_val,
#         phi=50, eta_j=eta_run, B=1200, beta=beta_local, price_model=price_model
#     )

#     return avg_cost(vb_prices), avg_cost(q_prices), avg_cost(r_prices)

# def paired_wilcoxon(vec_a, vec_b, label_a, label_b, scheme):
#     a = np.asarray(vec_a); b = np.asarray(vec_b)
#     mask = ~np.isnan(a) & ~np.isnan(b)
#     a = a[mask]; b = b[mask]
#     if len(a) == 0:
#         print(f"{scheme}: Not enough paired runs for {label_a} vs {label_b}.")
#         return
#     stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
#     diff = a - b
#     print(f"{scheme}: {label_a} vs {label_b} -> Wilcoxon W={stat}, p={p:.4g}, "
#           f"median Δ={np.median(diff):.3f} (Avg. Cost £/label) over n={len(a)} paired runs")

# # Number of runs (increase if you like)
# R = 50
# seeds = list(range(R))

# # Buyer-centric (BC)
# avgcost_BC_VBAL, avgcost_BC_QBCAL, avgcost_BC_RSC = [], [], []
# for s in seeds:
#     v, q, r = run_one_mse_split(s, price_model='BC')
#     avgcost_BC_VBAL.append(v); avgcost_BC_QBCAL.append(q); avgcost_BC_RSC.append(r)

# # Seller-centric (SC)
# avgcost_SC_VBAL, avgcost_SC_QBCAL, avgcost_SC_RSC = [], [], []
# for s in seeds:
#     v, q, r = run_one_mse_split(s, price_model='SC')
#     avgcost_SC_VBAL.append(v); avgcost_SC_QBCAL.append(q); avgcost_SC_RSC.append(r)

# # Paired tests (report medians as effect size)
# paired_wilcoxon(avgcost_BC_VBAL,  avgcost_BC_QBCAL, 'VBAL',  'QBCAL', 'BC')
# paired_wilcoxon(avgcost_BC_VBAL,  avgcost_BC_RSC,   'VBAL',  'RSC',   'BC')
# paired_wilcoxon(avgcost_BC_QBCAL, avgcost_BC_RSC,   'QBCAL', 'RSC',   'BC')

# paired_wilcoxon(avgcost_SC_VBAL,  avgcost_SC_QBCAL, 'VBAL',  'QBCAL', 'SC')
# paired_wilcoxon(avgcost_SC_VBAL,  avgcost_SC_RSC,   'VBAL',  'RSC',   'SC')
# paired_wilcoxon(avgcost_SC_QBCAL, avgcost_SC_RSC,   'QBCAL', 'RSC',   'SC')
