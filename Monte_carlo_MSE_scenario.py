# MSE-dependent scenario: figures, monte-carlo
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

def Algorithm(choose):
    # np.random.seed(seed)
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
def bootstrap_and_ambiguity(x_train, y_train, x_unlabeled, n_bootstrap):
    models, predictions = [], []
    n = x_train.shape[0]
    for i in range(n_bootstrap):
        model = LinearRegression()
        # np.random.seed(i + sampling_seed)
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
    Active learning process with support for buyer-centric (BC) or seller-centric (constant WTS) pricing models.
    
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

            # print(f"Iteration {iteration}: Purchased data: {data_bought_number}, Selected seller: {max_variance_index}, Price = {p_j}, New MSE = {new_mse}, MSE Reduction = {l_j}, Cumulative Budget = {cumulative_budget}")
            
            mse_list.append(new_mse)
            data_bought_number_list.append(data_bought_number)
            cumulative_budget_list.append(cumulative_budget)
            price_list.append(p_j)
            selected_sellers.append(max_variance_index)
        # else:
        #     # no purchase
        #     print(f"Iteration {iteration}: Data point skipped (WTP < price or no reduction), Current MSE = {previous_mse}, Price = {p_j}")
        
        # Remove the selected point from the unlabelled set
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

        # Terminate if the budget or target is reached
        if cumulative_budget >= B or new_mse <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            print(f" Variance reduction: {improve}")
  
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
    # np.random.seed(123)

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
                    # print(f"Iteration {iteration}:  p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f}), Skip.")
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
            # print(f"Iteration {iteration}:Data bought number: {data_bought_number}, Selected seller: {random_idx}, Price: {p_j:.4f}, Cumulative Budget: {cumulative_budget:.4f}, Variance: {new_mse:.6f}")

        else:
            # record unchanged variance 
            mse_list.append(previous_mse)
            # print(f"Iteration {iteration}: selected seller {random_idx}, Price: {p_j:.4f}, Cumulative budget: {cumulative_budget:.4f}, Variance (unchanged): {previous_mse:.6f}")
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
def qbc_active_learning(x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated, phi, B, beta, eta_j,price_model, n_bootstrap=10):
    _, variances = bootstrap_and_ambiguity(x_labelled, y_labelled, x_unlabelled, n_bootstrap)
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

        _, variances = bootstrap_and_ambiguity(x_labelled, y_labelled, x_unlabelled, n_bootstrap)
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
        # print(f"QBC - Iteration {iteration}:Purchased data: {data_bought_number}, Selected seller {max_variance_index},  New MSE = {new_mse}, MSE reduction = {l_j}, Price: {p_j:.4f}, Cumulative Budget = {cumulative_budget}")


        if cumulative_budget >= B or initial_mse - cumulative_mse_reduction <= beta:
            improve = (initial_mse - new_mse)/initial_mse
            print(f"  Variance reduction: {improve}")
            break

    return data_bought_number_list, mse_list, cumulative_budget_list, price_list, selected_sellers


# Load data
# data_seed = 123
choose2 = ['Hog_education_Madge', 'Hog_education_Rachael']
x_labelled, x_unlabelled, y_labelled, y_unlabelled, x_validated, y_validated = Algorithm(choose2)

# Example parameters
phi = 50
B = 1200
eta_j_list = np.full(x_unlabelled.shape[0],30)



def run_monte_carlo(strategy_function, num_iterations, choose, phi, B, eta_j_list, price_model):
    results = []
    for _ in range(num_iterations):
        # Reinitialize the data for each iteration
        x_labelled, x_unlabelled, y_labelled, y_unlabelled, x_validated, y_validated = Algorithm(choose)
        
        # Calculate initial MSE dynamically
        initial_model = LinearRegression().fit(x_labelled, y_labelled)
        initial_mse = fit_model_and_compute_mse(initial_model, x_validated, y_validated)
        beta = initial_mse * 0.8 
        
        # Call the active learning strategy
        data_bought_number_list, _, _, _, _ = strategy_function(
            x_labelled, y_labelled, x_unlabelled, y_unlabelled, x_validated, y_validated,
            phi=phi, B=B, beta=beta, eta_j=eta_j_list, price_model=price_model
        )
        
        # Remove the first 0 entry in data_bought_number_list
        data_bought_number_list = data_bought_number_list[1:]
        results.append(len(data_bought_number_list))  # Store total number of data points purchased
    
    return results


choose = ['Hog_education_Madge', 'Hog_education_Rachael']
num_iterations = 1000

vb_monte_carlo_results = run_monte_carlo(
    vb_active_learning, num_iterations, choose,
    phi=phi, B=B, eta_j_list=eta_j_list, price_model='BC'
)


rsc_monte_carlo_results = run_monte_carlo(
   random_sampling_corrected_strategy, num_iterations, choose,
    phi=phi, B=B, eta_j_list=eta_j_list, price_model='BC'
)

qbcal_monte_carlo_results = run_monte_carlo(
    qbc_active_learning, num_iterations, choose,
    phi=phi, B=B, eta_j_list=eta_j_list, price_model='BC'
)


def plot_monte_carlo_histogram(results, xlabel, ylabel, filename, ylim):
    plt.figure(figsize=(14, 12))

    # Determine the range and bins for the histogram
    min_value = min(results)
    max_value = max(results)
    bins = range(min_value, max_value + 2)  # Bins for integer values

    # Plot the histogram with bins
    counts, edges, patches = plt.hist(results, bins=bins, alpha=0.7, color='blue', edgecolor='black', align='left')

    # Filter x-ticks to only show bins with non-zero counts
    num_ticks = min(5, max_value - min_value + 1)  # Limit to 10 ticks at most
    xtick_indices = np.linspace(min_value, max_value, num=num_ticks, endpoint=True, dtype=int)
    plt.xticks(xtick_indices, fontsize=48)
    plt.yticks(fontsize=48)
    plt.ylim(ylim)


    # Set titles and labels
    plt.xlabel(xlabel, fontsize=50)
    plt.ylabel(ylabel, fontsize=50)
    plt.grid(False)
    ax = plt.gca()
    ax.spines ['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


plot_monte_carlo_histogram(
    vb_monte_carlo_results,
    # title="Monte Carlo Histogram of Purchased Data Points (VBAL)",
    xlabel="Number of purchased data points",
    ylabel="Frequency",
    filename="2_vb_monte_carlo_histogram.pdf",
    ylim = (0,200)
)

plot_monte_carlo_histogram(
    rsc_monte_carlo_results,
    # title="Monte Carlo Histogram of Purchased Data Points (RSC)",
    xlabel="Number of purchased data points",
    ylabel="Frequency",
    filename="2_rsc_monte_carlo_histogram.pdf",
    ylim = (0,100)
)

plot_monte_carlo_histogram(
    qbcal_monte_carlo_results,
    # title="Monte Carlo Histogram of Purchased Data Points (QBCAL)",
    xlabel="Number of purchased data points",
    ylabel="Frequency",
    filename="2_qbcal_monte_carlo_histogram.pdf",
    ylim = (0,200)
)

vb_monte_carlo_results = run_monte_carlo(
    vb_active_learning, num_iterations, choose,
    phi=phi, B=B, eta_j_list=eta_j_list, price_model='SC'
)

rsc_monte_carlo_results = run_monte_carlo(
    random_sampling_corrected_strategy, num_iterations, choose,
    phi=phi, B=B, eta_j_list=eta_j_list, price_model='SC'
)

qbcal_monte_carlo_results = run_monte_carlo(
    qbc_active_learning, num_iterations, choose,
    phi=phi, B=B, eta_j_list=eta_j_list, price_model='SC'
)

# Plotting
plot_monte_carlo_histogram(
    vb_monte_carlo_results,
    xlabel="Number of purchased data points",
    ylabel="Frequency",
    filename="2_vb_monte_carlo_histogram_sc.pdf",
    ylim = (0,200)
)

plot_monte_carlo_histogram(
    rsc_monte_carlo_results,
    xlabel="Number of purchased data points",
    ylabel="Frequency",
    filename="2_rsc_monte_carlo_histogram_sc.pdf",
    ylim = (0,100)

)

plot_monte_carlo_histogram(
    qbcal_monte_carlo_results,
    xlabel="Number of purchased data points",
    ylabel="Frequency",
    filename="2_qbcal_monte_carlo_histogram_sc.pdf",
    ylim = (0,200)
)
