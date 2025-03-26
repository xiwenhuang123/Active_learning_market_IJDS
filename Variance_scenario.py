"""
Variance-dependent scenario
@author: Xiwen Huang
"""

# Compare VBAL, QBCAL and RSC
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt


# fetch dataset 
real_estate_valuation = fetch_ucirepo(id=477) 
  
# data (as pandas dataframes) 
x = real_estate_valuation.data.features 
x = x.drop(columns=['X5 latitude', 'X6 longitude'])
y = real_estate_valuation.data.targets 



# Split the data into 30% labeled and 70% unlabelled
x_labelled, x_unlabelled, y_labelled, y_unlabelled = train_test_split(x, y, test_size=0.7, shuffle = True, random_state= 42)


x_labelled = x_labelled.to_numpy()
x_unlabelled = x_unlabelled.to_numpy()
y_labelled = y_labelled.to_numpy()
y_unlabelled = y_unlabelled.to_numpy()

# Function to calculate eta_j based on proximity to MRT station (simplified)
def generate_eta_j(x_unlabelled, feature_columns):
    # Scale eta_j based on proximity to the nearest MRT station (X3 distance to the nearest MRT station)
    eta_j = np.zeros(x_unlabelled.shape[0])

    # Normalize the 'X3 distance to the nearest MRT station' feature (3rd column in dataset)
    mrt_distance_col = feature_columns.index('X3 distance to the nearest MRT station')
    mrt_distances = x_unlabelled[:, mrt_distance_col]
    mrt_distances_normalized = (mrt_distances - mrt_distances.min()) / (mrt_distances.max() - mrt_distances.min())

    # Assign eta_j as inverse of distance (closer properties have higher eta_j)
    for i in range(len(x_unlabelled)):
        eta_j[i] = 0.1 + (1 - mrt_distances_normalized[i]) * 0.5 
    return eta_j


# Function to calculate the variance of coefficients on training data
def calculate_variance_of_coefficients_on_training(x, y):
    lr = LinearRegression()
    lr.fit(x, y)
    residuals = y - lr.predict(x)
    noise_estimate = np.std(residuals)
    xT_x_inv = np.linalg.inv(np.dot(x.T, x))
    coeff_variance = noise_estimate ** 2 * xT_x_inv
    return coeff_variance, noise_estimate

initial_variance, _ = calculate_variance_of_coefficients_on_training(x_labelled, y_labelled)
initial_variance = np.mean(np.diag(initial_variance))

def vb_active_learning(x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model, max_iterations=500):
    cumulative_var_reduction = 0
    cumulative_budget = 0
    data_bought_number = 0
    var_list = []
    data_bought_number_list = []
    cumulative_budget_list = []
    data_bought_number_list.append(data_bought_number) 
    cumulative_budget_list.append(cumulative_budget)
    price_list = []
    iteration = 0
    selected_sellers = []
    
    # Initial variance on the labeled training data
    initial_variance, _ = calculate_variance_of_coefficients_on_training(x_labelled, y_labelled)
    initial_variance = np.mean(np.diag(initial_variance))
    var_list.append(initial_variance)
    previous_variance = initial_variance

    print(f"VBAL: Initial Variance: {initial_variance:.6f} Budget Limit = {B}, Variance Threshold = {alpha}")

    while cumulative_budget < B and iteration < max_iterations:
        iteration += 1
        if x_unlabelled.shape[0] == 0:
            break

        # Calculate variances for each unlabelled data point
        _, noise_estimate = calculate_variance_of_coefficients_on_training(x_labelled, y_labelled)
        xT_x_inv = np.linalg.inv(np.dot(x_labelled.T, x_labelled))
        variances_new = np.array([noise_estimate * np.dot(np.dot(x_unlabelled[j], xT_x_inv), x_unlabelled[j].T) for j in range(x_unlabelled.shape[0])])
        max_variance_index = np.argmax(variances_new)
        x_max_var_point = x_unlabelled[max_variance_index].reshape(1, -1)
        y_max_var_point = y_unlabelled[max_variance_index]

        # Update the labeled set with the selected data point
        x_temp_labelled = np.vstack([x_labelled, x_max_var_point])
        y_temp_labelled = np.append(y_labelled, y_max_var_point)

        # Calculate the new variance after adding the data point
        new_variance, _ = calculate_variance_of_coefficients_on_training(x_temp_labelled, y_temp_labelled)
        new_variance = np.mean(np.diag(new_variance))
        
        variance_reduction = previous_variance - new_variance
        eta_j = eta_j_list[max_variance_index]

        # Willingness to sell and pay comparison
        if variance_reduction <= 0 or eta_j / variance_reduction > phi:
            print(f"Iteration {iteration}: insufficient variance reduction, skip.")
            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
            continue

        # Pricing model
        if price_model == 'BC':
            p_j = phi * variance_reduction
            if p_j <= eta_j:
                print(f"Iteration {iteration}:  p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f}), Skip.")
                x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
                y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
                continue
        else:
            p_j = eta_j

        # Update the labeled set and budget
        x_labelled = x_temp_labelled
        y_labelled = y_temp_labelled
        previous_variance = new_variance
        cumulative_var_reduction += variance_reduction
        cumulative_budget += p_j
        data_bought_number += 1
        data_bought_number_list.append(data_bought_number)

        # Logging the result of this iteration
        print(f"Iteration {iteration}: Data bought number: {data_bought_number}, Selected seller: {max_variance_index}, Variance Reduction: {variance_reduction:.6f}, New Variance: {new_variance:.6f}, Price: {p_j:.6f}, Cumulative Budget: {cumulative_budget:.6f}")

        # Record the results
        var_list.append(new_variance)
        cumulative_budget_list.append(cumulative_budget)
        price_list.append(p_j)
        selected_sellers.append(max_variance_index)

        # Check stopping criteria
        if cumulative_budget >= B or new_variance <= alpha:
            improve = (initial_variance - new_variance) / initial_variance
            print(f"Variance reduction: {improve}")
            break

        # Remove the selected data point from the unlabelled set
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers


# Random Sampling corrected Strategy 
def random_sampling_corrected_strategy(x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, eta_j, B, alpha, price_model):
    cumulative_var_reduction = 0
    cumulative_budget = 0
    var_list = []
    data_bought_number = 0
    data_bought_number_list = []
    cumulative_budget_list = []
    price_list = []
    iteration = 0
    selected_sellers = []

    var_list.append(initial_variance)
    print(f"RSC initial variance: {initial_variance:.6f},  Budget Limit = {B}, variance Threshold = {alpha}")
    data_bought_number_list.append(data_bought_number)
    cumulative_budget_list.append(cumulative_budget)
    previous_var = initial_variance

    # Set seed for reproducibility
    np.random. seed(123)

    while cumulative_budget < B and initial_variance - cumulative_var_reduction > alpha:
        iteration += 1
        
        # Randomly pick one data point (without any comparison)
        random_idx = np.random.randint(0, x_unlabelled.shape[0] - 1)
        x_random_point = x_unlabelled[random_idx].reshape(1, -1)
        y_random_point = y_unlabelled[random_idx]

        # Add it to the labelled set
        x_temp_labelled = np.vstack([x_labelled, x_random_point])
        y_temp_labelled = np.append(y_labelled, y_random_point)

        # Retrain the model with the new data
        new_var, _ = calculate_variance_of_coefficients_on_training(x_temp_labelled, y_temp_labelled)
        new_var = np.mean(np.diag(new_var))
        v_j = previous_var - new_var
        eta_j = eta_j_list[random_idx]

        # Pricing model
        if price_model == 'BC':
            if v_j > 0: 
                p_j = phi * v_j
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

        if v_j > 0:
            # Update labelled set and remove the selected point from the unlabelled set
            x_labelled = x_temp_labelled
            y_labelled = y_temp_labelled
            x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)

            # Update cumulative variables
            cumulative_var_reduction += v_j
            previous_var = new_var
            var_list.append(new_var)

            # Log details and update lists
            print(f"Iteration {iteration}:Data bought number: {data_bought_number}, Selected seller: {random_idx}, Price: {p_j:.4f}, Cumulative Budget: {cumulative_budget:.4f}, Variance: {new_var:.6f}")

        else:
            # record unchanged variance 
            var_list.append(previous_var)
            print(f"Iteration {iteration}: selected seller {random_idx}, Price: {p_j:.4f}, "
            f"Cumulative budget: {cumulative_budget:.4f}, Variance (unchanged): {previous_var:.6f}")
            x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)
            continue

        # Exit if the budget is exceeded or if the variance reduction is small enough
        if cumulative_budget >= B or new_var <= alpha:
            improve = (initial_variance - new_var)/initial_variance
            print(f"  Variance reduction: {improve}")
 
            break

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers


# Bootstrap Sampling Strategy
def bootstrap_and_ambiguity(x_train, y_train, x_unlabelled, n_bootstrap, sampling_seed):
    models = []
    predictions = []
    n = x_train.shape[0]
    for i in range(n_bootstrap):
        model = LinearRegression()
        np.random.seed(int(i + sampling_seed))
        train_idxs = np.random.choice(range(n), size=n, replace=True)
        x_train_bootstrap = x_train[train_idxs]
        y_train_bootstrap = y_train[train_idxs]
        model.fit(x_train_bootstrap, y_train_bootstrap)
        models.append(model)
        predictions.append(model.predict(x_unlabelled))
        variances = np.var(predictions, axis=0)
    return models, variances

# Bootstrap Sampling Strategy with v_j > 0 and eta_j / v_j < phi comparison
def qbc_active_learning(x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model, n_bootstrap=10, sampling_seed=42, max_iterations=500):
    cumulative_var_reduction = 0
    cumulative_budget = 0
    var_list = []
    data_bought_number = 0
    data_bought_number_list = []
    cumulative_budget_list = []
    price_list = []
    iteration = 0
    selected_sellers = []

    initial_var, _ = calculate_variance_of_coefficients_on_training(x_labelled, y_labelled)
    initial_var = np.mean(np.diag(initial_var))

    var_list.append(initial_var)
    print(f"QBCAL initial_variance: {initial_var:.6f}, Budget Limit = {B}, Variance Threshold = {alpha}")

    data_bought_number_list.append(data_bought_number)
    cumulative_budget_list.append(cumulative_budget)
    previous_var = initial_var

    while cumulative_budget < B and initial_var - cumulative_var_reduction > alpha:
        iteration += 1
        if iteration > max_iterations or x_unlabelled.shape[0] == 0:
            break

        # Create bootstrap models and calculate variance
        _, variances = bootstrap_and_ambiguity(x_labelled, y_labelled, x_unlabelled, n_bootstrap, sampling_seed)
        max_variance_index = np.argmax(variances)
        x_max_var_point = x_unlabelled[max_variance_index].reshape(1, -1)
        y_max_var_point = y_unlabelled[max_variance_index]

        # Get eta_j for the selected point
        eta_j = eta_j_list[max_variance_index]

        # Temporarily add the selected data point to the labelled set
        x_temp_labelled = np.vstack([x_labelled, x_max_var_point])
        y_temp_labelled = np.append(y_labelled, y_max_var_point)

        # Retrain the model with the new data point
        new_var, _ = calculate_variance_of_coefficients_on_training(x_temp_labelled, y_temp_labelled)
        new_var = np.mean(np.diag(new_var))
        v_j = previous_var - new_var


        # Only proceed if v_j > 0 and WTP allows
        if price_model == 'BC':
            p_j = phi * v_j
            if p_j <= eta_j:
                # Add a fallback mechanism for small p_j
                print(f"Skipping point: p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f})")
                x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
                y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
                continue
        else:  # price_model == 'SC'
            p_j = eta_j

        # Only proceed if v_j > 0 and WTP allows
        if v_j > 0 and eta_j/v_j < phi:
            # Update labelled set and budget
            x_labelled = x_temp_labelled
            y_labelled = y_temp_labelled
            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

            cumulative_var_reduction += v_j
            cumulative_budget += p_j

            # Update tracking variables
            previous_var = new_var
            var_list.append(new_var)
            data_bought_number += 1
            data_bought_number_list.append(data_bought_number)
            cumulative_budget_list.append(cumulative_budget)
            price_list.append(p_j)
            selected_sellers.append(max_variance_index)

            print(f"Iteration {iteration}: Data bought: {data_bought_number}, Seller: {max_variance_index}, Price: {p_j:.4f}, Budget: {cumulative_budget:.4f}, Variance: {new_var:.6f}")

        # Check stopping condition
        if cumulative_budget >= B or new_var <= alpha:
            print(f"Stopping criteria met. Variance reduction: {(initial_var - new_var) / initial_var:.4%}")
            break

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers



def plot_variance_reduction_comparison(vb_data_bought, vb_var_list, rsc_data_bought, rsc_var_list, qbc_data_bought, qbc_var_list, filename):
    plt.figure(figsize=(14, 12))
    plt.plot(vb_data_bought, vb_var_list, marker='o', label='VBAL', color='#d62728', linewidth=3, markersize=10)
    plt.plot(qbc_data_bought, qbc_var_list, marker='s', label='QBCAL', color='#2ca02c', linewidth=3, markersize=10)
    plt.plot(rsc_data_bought, rsc_var_list, marker='x', linestyle='-.', label='RSC (Baseline)', color='#1f77b4', linewidth=3, markersize=10)

    plt.xlabel('Data points purchased', fontsize=40)
    max_purchase = max(len(vb_data_bought), len(rsc_data_bought), len(qbc_data_bought))
    plt.xticks(ticks=list(range(1, max_purchase, 2)), fontsize=35)
    plt.ylabel('Variance', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.text(len(rsc_data_bought) * 0.1, alpha-0.0005 , f'Variance threshold = {alpha:.3f}', color='red', fontsize=35, weight='bold')
    plt.axhline(y=alpha, color='red', linestyle='--')
    plt.legend(fontsize=30)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


def plot_budget_utilization_comparison(vb_data_bought, vb_budget_list,rsc_data_bought, rsc_budget_list, qbc_data_bought, qbc_budget_list, filename):
    plt.figure(figsize=(14, 12))
    plt.plot(vb_data_bought, vb_budget_list, marker='o', label='VBAL', color='#d62728', linewidth=3, markersize=10)
    plt.plot(qbc_data_bought, qbc_budget_list, marker='s', label='QBCAL', color='#2ca02c', linewidth=3, markersize=10)
    plt.plot(rsc_data_bought, rsc_budget_list, marker = 'x',linestyle='-.',  label='RSC (baseline)', color = '#1f77b4', linewidth=3, markersize=10)

    plt.axhline(y=B, color='red', linestyle='--')
    plt.text(len(rsc_data_bought) * 0.2, B+0.15 , f'Budget Limit = {B}', color='red', fontsize=35, weight='bold')
    max_purchase = max(len(vb_data_bought), len(rsc_data_bought), len(qbc_data_bought))
    plt.xlabel('Data points purchased', fontsize=40)
    plt.xticks(ticks=list(range(1, max_purchase, 2)), fontsize=35)
    plt.ylabel('Cumulative budget [£]', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=30)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



# Define the feature columns used in the dataset
feature_columns = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']

# Generate eta_j values based on distance to MRT only
eta_j_list = generate_eta_j(x_unlabelled, feature_columns) 

# Example execution for three strategies
phi = 1200
B = 15
alpha = initial_variance * 0.8
n_bootstrap = 5
sampling_seed = 42
# Call each strategy with given parameters to obtain the required variables for plotting
vb_BC_data_bought, vb_BC_var_list, vb_BC_budget_list, vb_BC_price_list, vb_BC_sellers = vb_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='BC'
)

vb_SC_data_bought, vb_SC_var_list, vb_SC_budget_list, vb_SC_price_list, vb_SC_sellers = vb_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='SC'
)

rsc_BC_data_bought, rsc_BC_var_list, rsc_BC_budget_list, rsc_BC_price_list, rsc_BC_sellers = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, eta_j_list, B, alpha, price_model='BC'
)

rsc_SC_data_bought, rsc_SC_var_list, rsc_SC_budget_list, rsc_SC_price_list, rsc_SC_sellers  = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, eta_j_list, B, alpha, price_model='SC'
)

qbc_BC_data_bought, qbc_BC_var_list, qbc_BC_budget_list, qbc_BC_price_list, qbc_BC_sellers = qbc_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='BC', n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
)

qbc_SC_data_bought, qbc_SC_var_list, qbc_SC_budget_list, qbc_SC_price_list, qbc_SC_sellers = qbc_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='SC', n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
)



# Plotting Variance Reduction Comparisons
plot_variance_reduction_comparison(
    vb_BC_data_bought, vb_BC_var_list,  rsc_BC_data_bought, rsc_BC_var_list, qbc_BC_data_bought, qbc_BC_var_list,
    filename='variance_reduction_comparison_BC.pdf'
)

plot_variance_reduction_comparison(
    vb_SC_data_bought, vb_SC_var_list, rsc_SC_data_bought, rsc_SC_var_list, qbc_SC_data_bought, qbc_SC_var_list,
    filename='variance_reduction_comparison_SC.pdf'
)

# Plotting Budget Utilization Comparisons
plot_budget_utilization_comparison(
    vb_BC_data_bought, vb_BC_budget_list,  rsc_BC_data_bought, rsc_BC_budget_list, qbc_BC_data_bought, qbc_BC_budget_list,
    filename='budget_utilization_comparison_BC.pdf'
)

plot_budget_utilization_comparison(
    vb_SC_data_bought, vb_SC_budget_list,  rsc_SC_data_bought, rsc_SC_budget_list, qbc_SC_data_bought, qbc_SC_budget_list,
    filename='budget_utilization_comparison_SC.pdf'
)



def plot_percentage_variance_reduction_comparison(
    vb_price_list, vb_variance_list,  
    rsc_price_list, rsc_variance_list, qbc_price_list, qbc_variance_list, filename
):
    """
    Plot Δv/Δc (variance reduction per unit cost) against data purchase number.

    Parameters:
    - vb_price_list: List of VBAL prices.
    - vb_variance_list: List of VBAL variance reductions.
    - rs_price_list: List of RS prices.
    - rs_variance_list: List of RS variance reductions.
    - rsc_price_list: List of RSC prices.
    - rsc_variance_list: List of RSC variance reductions.
    - qbc_price_list: List of QBCAL prices.
    - qbc_variance_list: List of QBCAL variance reductions.
    """
    plt.figure(figsize=(14, 12))

    # Prepare data
    strategies = [
        ("VBAL", vb_price_list, vb_variance_list),
        ("RSC", rsc_price_list, rsc_variance_list),
        ("QBCAL", qbc_price_list, qbc_variance_list),
    ]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']  # Colors per strategy
    markers = ['o', 'x', '^', 's']  # Markers per strategy

    for (strategy_name, price_list, variance_list), color, marker in zip(strategies, colors, markers):
        # Calculate Δv and efficiency (Δv/Δc)
        delta_v = [variance_list[i - 1] - variance_list[i] for i in range(1, len(variance_list))]
        delta_c = price_list[0:]  # Cost is taken directly from the price list (from index 1 onward)
        efficiency = [v / c if c != 0 else 0 for v, c in zip(delta_v, delta_c)]

        # Plot Δ/Δc vs data purchase number
        purchase_numbers = list(range(1, len(efficiency) + 1))
        plt.plot(purchase_numbers, efficiency, 
                 marker=marker, label=strategy_name, 
                 color=color, linewidth=3, markersize=10)

    # Labels and legends
    plt.xlabel('Data point purchased', fontsize=40)
    plt.ylabel('Δv/Δc', fontsize=40)
    max_purchase = max(len(vb_variance_list), len(rsc_variance_list), len(qbc_variance_list))
    plt.xticks(ticks=list(range(1, max_purchase, 2)), fontsize=35)
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
    vb_BC_price_list, vb_BC_var_list,
    rsc_BC_price_list, rsc_BC_var_list,
    qbc_BC_price_list, qbc_BC_var_list,
    filename = "1_BC_v_c_vs_purchase_number.pdf"
)

plot_percentage_variance_reduction_comparison(
    vb_SC_price_list, vb_SC_var_list,
    rsc_SC_price_list, rsc_SC_var_list,
    qbc_SC_price_list, qbc_SC_var_list,
    filename = "1_SC_v_c_vs_purchase_number.pdf"
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
    xtick_interval = max(1, len(aligned_bc_prices) // 10)  # Show a tick for every 10% of total sellers
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
    bc_price_list=vb_BC_price_list, 
    sc_price_list=vb_SC_price_list, 
    bc_sellers = vb_BC_sellers,
    sc_sellers = vb_SC_sellers,
    strategy_name="VBAL", 
    filename="1_price_bar_chart_vbal_bc_sc.pdf", 
    global_min_y=global_min_y, 
    global_max_y=global_max_y
)

# QBCAL
plot_sorted_price_bar_chart_with_sc(
    bc_price_list=qbc_BC_price_list, 
    sc_price_list=qbc_SC_price_list, 
    bc_sellers = qbc_BC_sellers,
    sc_sellers = qbc_SC_sellers,
    strategy_name="QBCAL", 
    filename="1_price_bar_chart_qbcal_bc_sc.pdf", 
    global_min_y=global_min_y, 
    global_max_y=global_max_y
)

# RSC
plot_sorted_price_bar_chart_with_sc(
    bc_price_list=rsc_BC_price_list, 
    sc_price_list=rsc_SC_price_list, \
    bc_sellers = rsc_BC_sellers,
    sc_sellers = rsc_SC_sellers,
    strategy_name="RSC", 
    filename="1_price_bar_chart_rsc_bc_sc.pdf", 
    global_min_y=global_min_y, 
    global_max_y=global_max_y
)
