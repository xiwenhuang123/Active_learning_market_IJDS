"""
Estimation-quality-focused scenario (real-estate valuation case study).
Author: Xiwen Huang

This script reproduces the following results in the paper:
- Figures 7
- Figures 8

"""
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ============================ Paths & timer ============================
start_time = time.time()
if '__file__' not in globals():
    __file__ = str(Path.cwd() / "IJDS_Variance.py")

THIS_DIR = Path(__file__).resolve().parent
PLOT_DIR = THIS_DIR / "Plots" / "VarianceScenario"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

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

        # bad case: no useful or not worth it
        if variance_reduction <= 0 or eta_j / variance_reduction > phi:
            if price_model == 'BC':
                p_j = 0.0
            else:   # SC
                p_j = eta_j

            cumulative_budget += p_j   # always pay something in SC, pay 0 in BC
            cumulative_budget_list.append(cumulative_budget)
            price_list.append(p_j)
            var_list.append(previous_variance)
            data_bought_number+=1
            data_bought_number_list.append(data_bought_number)

            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
            continue

        # good case: variance_reduction > 0 and eta_j / variance_reduction <= phi
        if price_model == 'BC':
            p_j = phi * variance_reduction
        else:
            p_j = eta_j

        # update labelled set and budget
        x_labelled = x_temp_labelled
        y_labelled = y_temp_labelled
        previous_variance = new_variance
        cumulative_var_reduction += variance_reduction

        cumulative_budget += p_j
        cumulative_budget_list.append(cumulative_budget)
        data_bought_number += 1
        data_bought_number_list.append(data_bought_number)
        price_list.append(p_j)
        selected_sellers.append(max_variance_index)

        print(
            f"Iteration {iteration}: Data bought number: {data_bought_number}, "
            f"Selected seller: {max_variance_index}, Variance Reduction: {variance_reduction:.6f}, "
            f"New Variance: {new_variance:.6f}, Price: {p_j:.6f}, "
            f"Cumulative Budget: {cumulative_budget:.6f}"
        )

        var_list.append(new_variance)


        # Check stopping criteria
        if cumulative_budget >= B or new_variance <= alpha:
            print(f"VBAL: Variance reduction: {(initial_variance- new_variance) / initial_variance:.4%}")
            break

        # Remove the selected data point from the unlabelled set
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers

# Random Sampling corrected Strategy 
def random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled,
    phi, B, alpha, eta_j_list, price_model, rsc_seed=None
):
    # -------------------------
    # Initialize
    # -------------------------
    if rsc_seed is not None:
        np.random.seed(rsc_seed)

    # Compute local initial variance (avoid using global!)
    initial_var_mat, _ = calculate_variance_of_coefficients_on_training(x_labelled, y_labelled)
    initial_var = np.mean(np.diag(initial_var_mat))

    cumulative_var_reduction = 0
    cumulative_budget = 0

    var_list = [initial_var]
    data_bought_number = 0
    data_bought_number_list = [0]
    cumulative_budget_list = [0]
    price_list = []
    selected_sellers = []

    previous_var = initial_var

    print(f"RSC initial variance: {initial_var:.6f}, Budget Limit = {B}, Variance Threshold = {alpha}")

    # -------------------------
    # Main loop
    # -------------------------
    while cumulative_budget < B and (initial_var - cumulative_var_reduction) > alpha:

        if len(x_unlabelled) == 0:
            break

        # Randomly select a sample
        j = np.random.randint(0, x_unlabelled.shape[0])

        x_new = x_unlabelled[j].reshape(1, -1)
        y_new = y_unlabelled[j]

        # Temporarily add to labelled set
        x_temp = np.vstack([x_labelled, x_new])
        y_temp = np.append(y_labelled, y_new)

        # Compute new variance
        new_var_mat, _ = calculate_variance_of_coefficients_on_training(x_temp, y_temp)
        new_var = np.mean(np.diag(new_var_mat))

        v_j = previous_var - new_var
        eta_j = eta_j_list[j]

        # -------------------------
        # Price computation
        # -------------------------
        if price_model == "BC":
            p_j = phi * v_j if v_j > 0 else 0
        else:
            p_j = eta_j

        cumulative_budget += p_j
        cumulative_budget_list.append(cumulative_budget)
        price_list.append(p_j)

        data_bought_number += 1
        data_bought_number_list.append(data_bought_number)

        # -------------------------
        # Accept or reject the point
        # -------------------------
        if v_j > 0:
            # Accept the data point
            x_labelled = x_temp
            y_labelled = y_temp

            x_unlabelled = np.delete(x_unlabelled, j, axis=0)
            y_unlabelled = np.delete(y_unlabelled, j, axis=0)

            previous_var = new_var
            cumulative_var_reduction += v_j
            selected_sellers.append(j)

            var_list.append(new_var)

        else:
            # Reject it (no update)
            var_list.append(previous_var)

            x_unlabelled = np.delete(x_unlabelled, j, axis=0)
            y_unlabelled = np.delete(y_unlabelled, j, axis=0)

        if cumulative_budget >= B or new_var <= alpha:
            break

    return (
        data_bought_number_list,
        var_list,
        cumulative_budget_list,
        price_list,
        selected_sellers
    )

# Bootstrap Sampling Strategy
def bootstrap_and_ambiguity(x_train, y_train, x_unlabelled, n_bootstrap):
    models = []
    predictions = []
    n = x_train.shape[0]
    for i in range(n_bootstrap):
        model = LinearRegression()
        # np.random.seed(int(i + sampling_seed))
        train_idxs = np.random.choice(range(n), size=n, replace=True)
        x_train_bootstrap = x_train[train_idxs]
        y_train_bootstrap = y_train[train_idxs]
        model.fit(x_train_bootstrap, y_train_bootstrap)
        models.append(model)
        predictions.append(model.predict(x_unlabelled))
    variances = np.var(predictions, axis=0)
    return models, variances

# Bootstrap Sampling Strategy with v_j > 0 and eta_j / v_j < phi comparison
def qbc_active_learning(x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model, n_bootstrap=10, max_iterations=500):
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
        _, variances = bootstrap_and_ambiguity(x_labelled, y_labelled, x_unlabelled, n_bootstrap)
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
        if v_j <= 0 or eta_j / v_j > phi:
            # bad case
            if price_model == 'BC':
                p_j = 0
            else:   # SC
                p_j = eta_j

            cumulative_budget += p_j   # always pay something in SC, pay 0 in BC]
            cumulative_budget_list.append(cumulative_budget)
            price_list.append(p_j)
            data_bought_number += 1
            data_bought_number_list.append(data_bought_number)
            var_list.append(previous_var)


            # DO NOT add to x_labelled, y_labelled
            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
            continue

        # Only proceed if v_j > 0 and WTP allows
        if v_j > 0 and eta_j/v_j < phi:
            if price_model == 'BC':
                p_j = phi * v_j
            else:
                p_j = eta_j
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
            print(f"QBCAL: Variance reduction: {(initial_var - new_var) / initial_var:.4%}")
            break

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers

# Define the feature columns used in the dataset
feature_columns = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']

# Generate eta_j values based on distance to MRT only
eta_j_list = generate_eta_j(x_unlabelled, feature_columns) 


# Example execution for three strategies
phi = 1200
B = 15

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
    always under Plots/VarianceScenario.
    """
    fig.set_size_inches(width, height)
    out_path = PLOT_DIR / filename  
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0.1
    )


def run_monte_carlo(strategy_function, num_iterations, x, y, phi, B, price_model, test_size=0.7):
    results = []
    for i in range(num_iterations):
        np.random.seed(i) 

        # Data splitting
        x_labelled, x_unlabelled, y_labelled, y_unlabelled = train_test_split(
            x, y, test_size=test_size, random_state=i
        )

        # Convert to numpy arrays and flatten targets
        x_labelled = x_labelled.to_numpy()
        x_unlabelled = x_unlabelled.to_numpy()
        y_labelled = y_labelled.to_numpy().flatten()
        y_unlabelled = y_unlabelled.to_numpy().flatten()

        eta_j_list_iter = generate_eta_j(x_unlabelled, feature_columns)

          # Dynamically calculate initial alpha
        initial_var, _ = calculate_variance_of_coefficients_on_training(x_labelled, y_labelled)
        initial_var = np.mean(np.diag(initial_var))  # Extract the scalar variance
        alpha = initial_var * 0.8 

        data_bought_number_list, _, _, _, selected_sellers= strategy_function(
            x_labelled, y_labelled, x_unlabelled, y_unlabelled,
            phi=phi, B=B, alpha=alpha, eta_j_list=eta_j_list_iter, price_model=price_model
        )

        results.append(len(selected_sellers))  # Store total number of data points purchased

    return results


def plot_monte_carlo_histogram(results, xlabel, ylabel, filename, ylim):
    # Determine the range and bins for the histogram
    fig, ax = plt.subplots(figsize=(14, 12))

    # Determine histogram range and bins
    min_value = int(min(results))
    max_value = int(max(results))
    bins = range(min_value, max_value + 2)  # bins for integer values

    # Plot histogram
    counts, edges, patches = ax.hist(
        results, bins=bins, alpha=0.7, color='blue', edgecolor='black', align='left'
    )

    # Configure x-ticks (limit to ~5 to avoid overlap)
    num_ticks = min(5, max_value - min_value + 1)
    xtick_indices = np.linspace(min_value, max_value, num=num_ticks, endpoint=True, dtype=int)
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([str(x) for x in xtick_indices], fontsize=48)

    # Configure y-ticks and limits
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=48)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=48)

    # Labels
    ax.set_xlabel(xlabel, fontsize=50)
    ax.set_ylabel(ylabel, fontsize=50)

    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.grid(False)

    # Add arrowheads
    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()



# Monte Carlo simulation for each strategy
num_iterations = 1000


# VBAL Monte Carlo
vb_monte_carlo_results = run_monte_carlo(
    vb_active_learning, num_iterations, x, y,
    phi=phi, B=B,  price_model='BC'
)

# RSC Monte Carlo
rsc_monte_carlo_results = run_monte_carlo(
    random_sampling_corrected_strategy, num_iterations, x, y,
    phi=phi, B=B ,price_model='BC'
)

# QBCAL Monte Carlo
qbcal_monte_carlo_results = run_monte_carlo(
    qbc_active_learning, num_iterations, x, y,
    phi=phi, B=B, price_model='BC'
)


plot_monte_carlo_histogram(
    vb_monte_carlo_results,
    # title="Monte Carlo Histogram of Purchased Data Points (VBAL)",
    xlabel="Effectively purchased data points",
    ylabel="Frequency",
    filename="1_vb_monte_carlo_histogram.pdf",
    ylim = (0,400)
)


plot_monte_carlo_histogram(
    rsc_monte_carlo_results,
    # title="Monte Carlo Histogram of Purchased Data Points (RSC)",
    xlabel="Effectively purchased data points",
    ylabel="Frequency",
    filename="1_rsc_monte_carlo_histogram.pdf",
    ylim = (0,400)
)

plot_monte_carlo_histogram(
    qbcal_monte_carlo_results,
    # title="Monte Carlo Histogram of Purchased Data Points (QBCAL)",
    xlabel="Effectively purchased data points",
    ylabel="Frequency",
    filename="1_qbcal_monte_carlo_histogram.pdf",
    ylim = (0,400)
)

vb_monte_carlo_results_sc = run_monte_carlo(
    vb_active_learning, num_iterations, x, y,
    phi=phi, B=B, price_model='SC'
)


# RSC Monte Carlo
rsc_monte_carlo_results_sc = run_monte_carlo(
    random_sampling_corrected_strategy, num_iterations, x, y,
    phi=phi, B=B, price_model='SC'
)

# QBCAL Monte Carlo
qbcal_monte_carlo_results_sc = run_monte_carlo(
    qbc_active_learning, num_iterations, x, y,
    phi=phi, B=B, price_model='SC'
)

plot_monte_carlo_histogram(
    vb_monte_carlo_results_sc,
    # title="Monte Carlo Histogram of Purchased Data Points (VBAL)",
    xlabel="Effectively purchased data points",
    ylabel="Frequency",
    filename="1_vb_monte_carlo_histogram_SC.pdf",
    ylim = (0,400)
)

plot_monte_carlo_histogram(
    rsc_monte_carlo_results_sc,
    # title="Monte Carlo Histogram of Purchased Data Points (RSC)",
    xlabel="Effectively purchased data points",
    ylabel="Frequency",
    filename="1_rsc_monte_carlo_histogram_SC.pdf",
    ylim = (0,400)
)

plot_monte_carlo_histogram(
    qbcal_monte_carlo_results_sc,
    # title="Monte Carlo Histogram of Purchased Data Points (QBCAL)",
    xlabel="Effectively purchased data points",
    ylabel="Frequency",
    filename="1_qbcal_monte_carlo_histogram_SC.pdf",
    ylim = (0,400)
)

# ============================ Total runtime ============================
end_time = time.time()
print("\n===================================")
print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")
print("===================================\n")
