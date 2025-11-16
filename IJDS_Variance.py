"""
Estimation-quality-focused scenario (real-estate valuation case study).
Author: Xiwen Huang

This script reproduces the following results in the paper:
- Figures 3a–3b
- Figures 4a–4b
- Figure 5
- Figure 6
- Figures 9a–9f
- Table 2
- Tables 4 and 5 (Appendix)

It also generates additional diagnostic plots that are not shown in the paper.
"""

# Compare VBAL, QBCAL and RSC
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axisartist.axislines import AxesZero
from mpl_toolkits.axisartist import SubplotZero
from pathlib import Path

# Base directories (repo-relative, reproducible on any machine)
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

# Function to calculate eta_j based on proximity to MRT station 
def generate_eta_j(x_unlabelled, feature_columns, offset, coeff):
    """
    Compute η_j for each unlabelled data point based on distance to the nearest MRT station.
    η_j = offset + (1 - normalized_distance) * coeff

    Parameters
    ----------
    x_unlabelled : np.ndarray
        Unlabelled feature matrix.
    feature_columns : list of str
        Column names in the same order as x_unlabelled.
    offset : float, optional
        Minimum η_j (default = 0.1).
    coeff : float, optional
        Scaling factor controlling how much η_j changes with distance (default = 0.5).

    Returns
    -------
    eta_j : np.ndarray
        Array of willingness-to-sell values.
    """
    eta_j = np.zeros(x_unlabelled.shape[0])
    mrt_col = feature_columns.index('X3 distance to the nearest MRT station')
    mrt_distances = x_unlabelled[:, mrt_col]
    den = mrt_distances.max() - mrt_distances.min()
    mrt_norm = (mrt_distances - mrt_distances.min()) / den if den != 0 else np.zeros_like(mrt_distances)
    eta_j = offset + (1 - mrt_norm) * coeff
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

# VBAL (variance-based active learning strategy) 
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
            # print(f"Iteration {iteration}: insufficient variance reduction, skip.")
            x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
            y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)
            continue

        # Pricing model
        if price_model == 'BC':
            p_j = phi * variance_reduction
            if p_j <= eta_j:
                # print(f"Iteration {iteration}:  p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f}), Skip.")
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
            print(f"VBAL: Variance reduction: {(initial_variance- new_variance) / initial_variance:.4%}")
            break

        # Remove the selected data point from the unlabelled set
        x_unlabelled = np.delete(x_unlabelled, max_variance_index, axis=0)
        y_unlabelled = np.delete(y_unlabelled, max_variance_index, axis=0)

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers


# Random sampling corrected strategy 
def random_sampling_corrected_strategy(x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, eta_j, B, alpha, price_model, rsc_seed):
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
    np.random. seed(rsc_seed)

    while cumulative_budget < B and initial_variance - cumulative_var_reduction > alpha:
        iteration += 1

        n = x_unlabelled.shape[0]
        if n == 0:
            break

        
        # Randomly pick one data point (without any comparison)
        random_idx = np.random.randint(0, x_unlabelled.shape[0])
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
            # print(f"Iteration {iteration}: selected seller {random_idx}, Price: {p_j:.4f}, "
            # f"Cumulative budget: {cumulative_budget:.4f}, Variance (unchanged): {previous_var:.6f}")
            x_unlabelled = np.delete(x_unlabelled, random_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, random_idx, axis=0)
            continue

        # Exit if the budget is exceeded or if the variance reduction is small enough
        if cumulative_budget >= B or new_var <= alpha:
            print(f" RSC: Variance reduction: {(initial_variance - new_var) / initial_variance:.4%}")
 
            break

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers


# Bootstrap Sampling function in QBCAL
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

# QBCAL (Query-by-committee acite learning strategy)
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
                # print(f"Skipping point: p_j ({p_j:.6f}) <= eta_j ({eta_j:.6f})")
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
            print(f"QBCAL: Variance reduction: {(initial_var - new_var) / initial_var:.4%}")
            break

    return data_bought_number_list, var_list, cumulative_budget_list, price_list, selected_sellers



# add arrows and ticks for figures
def _auto_xticks_from_lists(list1, list2, list3, num_ticks=6):
    """Generate ~num_ticks evenly spaced integer ticks starting at 1, ending at max N."""
    max_purchase = max(len(list1), len(list2), len(list3))
    if max_purchase <= 1:
        return [1]

    step = max(1, round(max_purchase / (num_ticks - 1)))
    xticks = list(range(1, max_purchase + 1, step))
    if xticks[-1] != max_purchase:
        xticks.append(max_purchase)
    return xticks, max_purchase

def _add_axis_arrows(ax, lw=2.5):
    """
    Draws arrowheads for x→ and y↑ using axes-fraction coordinates.
    Keeps bottom/left spines so ticks align with axes.
    Arrow length/position is visually consistent across plots.
    """
    # x-axis arrow (→)
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


def compute_improvement_list(var_list):
    """
    Compute cumulative model improvement (%) given variance list.
    Smaller variance means better performance.
    """
    base_value = var_list[0]
    return [(base_value - v) / base_value * 100 for v in var_list]

# ======================== Generates Figures 3a and 4a in the paper ========================
def plot_model_improvement_comparison(
    vb_data_bought, vb_var_list,
    rsc_data_bought, rsc_var_list,
    qbc_data_bought, qbc_var_list,
    filename="model_improvement_comparison.pdf"
):
    
    vb_improve_list  = compute_improvement_list(vb_var_list)
    rsc_improve_list = compute_improvement_list(rsc_var_list)
    qbc_improve_list = compute_improvement_list(qbc_var_list)

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(vb_data_bought, vb_improve_list,
            marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    ax.plot(qbc_data_bought, qbc_improve_list,
            marker='s', label='QBCAL', color='#2ca02c', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#2ca02c', markeredgecolor='white')    
    ax.plot(rsc_data_bought, rsc_improve_list,
            marker='^', label='RSC', color='#1f77b4',
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4',markeredgecolor='white')

    max_purchase = max(len(vb_data_bought), len(rsc_data_bought), len(qbc_data_bought))
    step = max(1, max_purchase // 5)
    ax.set_xticks(range(1, max_purchase + 1, step))

    ax.set_xlabel('Data points purchased', fontsize=35)
    ax.set_ylabel('Model improvement [%]', fontsize=35)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)

    threshold = 20
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
    ax.text(
        max_purchase * 0.09,
        threshold +2,
        f'Improvement threshold = {threshold:.0f}%',
        color='red', fontsize=30, weight='bold'
    )

    ax.legend(fontsize=30, loc='lower right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    _add_axis_arrows(ax, lw=2.5)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()

# ======================== Produces a reference-only variance reduction plot for Figures 3a and 4a (not included in the paper) ========================

def plot_variance_reduction_comparison(
    vb_data_bought, vb_var_list,
    rsc_data_bought, rsc_var_list,
    qbc_data_bought, qbc_var_list,
    filename
):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.plot(vb_data_bought, vb_var_list,
            marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    ax.plot(qbc_data_bought, qbc_var_list,
             marker='s', label='QBCAL', color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')
    ax.plot(rsc_data_bought, rsc_var_list,
            marker='^', label='RSC', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')
    xticks, max_purchase = _auto_xticks_from_lists(
        vb_data_bought, rsc_data_bought, qbc_data_bought, num_ticks=6
    )
    ax.set_xticks(xticks)
    ax.set_xlabel('Data points purchased', fontsize=35)
    ax.set_ylabel(r'Variance [ TWD$^2$]', fontsize=35)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2)
    ax.text(
        max_purchase * 0.1,          
        alpha - 0.0005,              
        f'Variance threshold = {alpha:.3f}',
        color='red', fontsize=30, weight='bold'
    )
    ax.legend(
        fontsize=30,
        loc='upper right'
    )
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()



# ======================== Generates Figures 3b and 4b in the paper ========================
def plot_budget_utilization_comparison(
    vb_data_bought, vb_budget_list,
    rsc_data_bought, rsc_budget_list,
    qbc_data_bought, qbc_budget_list,
    filename
):
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(vb_data_bought, vb_budget_list,
            marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    ax.plot(qbc_data_bought, qbc_budget_list,
            marker='s', label='QBCAL', color='#2ca02c',
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')
    ax.plot(rsc_data_bought, rsc_budget_list,
            marker='^', label='RSC', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')

    # --- X ticks ---
    xticks, max_purchase = _auto_xticks_from_lists(
        vb_data_bought, rsc_data_bought, qbc_data_bought, num_ticks=6
    )
    ax.set_xticks(xticks)

    ax.axhline(y=B, color='red', linestyle='--', linewidth=2)
    ax.text(
        max_purchase * 0.2,
        B + 0.15,
        f'Budget Limit = {B}',
        color='red', fontsize=30, weight='bold'
    )

    ax.set_xlabel('Data points purchased', fontsize=35)
    ax.set_ylabel('Cumulative budget [£]', fontsize=35)

    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)

    ax.legend(
        fontsize=30,
        loc='lower right',   
    )

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    _add_axis_arrows(ax, lw=2.5)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()





# Define the feature columns used in the dataset
feature_columns = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']

# Generate eta_j values based on distance to MRT only
eta_j_list = generate_eta_j(x_unlabelled, feature_columns, offset=0.1, coeff=0.5) 

# Example execution for three strategies
phi = 1200
B = 15
alpha = initial_variance * 0.8
n_bootstrap = 5
sampling_seed = 42
rsc_seed = 123
# Call each strategy with given parameters to obtain the required variables for plotting
vb_BC_data_bought, vb_BC_var_list, vb_BC_budget_list, vb_BC_price_list, vb_BC_sellers = vb_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='BC'
)

vb_SC_data_bought, vb_SC_var_list, vb_SC_budget_list, vb_SC_price_list, vb_SC_sellers = vb_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='SC'
)

rsc_BC_data_bought, rsc_BC_var_list, rsc_BC_budget_list, rsc_BC_price_list, rsc_BC_sellers = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, eta_j_list, B, alpha, price_model='BC', rsc_seed = rsc_seed
)

rsc_SC_data_bought, rsc_SC_var_list, rsc_SC_budget_list, rsc_SC_price_list, rsc_SC_sellers  = random_sampling_corrected_strategy(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, eta_j_list, B, alpha, price_model='SC', rsc_seed = rsc_seed
)

qbc_BC_data_bought, qbc_BC_var_list, qbc_BC_budget_list, qbc_BC_price_list, qbc_BC_sellers = qbc_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='BC', n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
)

qbc_SC_data_bought, qbc_SC_var_list, qbc_SC_budget_list, qbc_SC_price_list, qbc_SC_sellers = qbc_active_learning(
    x_labelled, y_labelled, x_unlabelled, y_unlabelled, phi, B, alpha, eta_j_list, price_model='SC', n_bootstrap=n_bootstrap, sampling_seed=sampling_seed
)


plot_model_improvement_comparison(
    vb_BC_data_bought, vb_BC_var_list,
    rsc_BC_data_bought, rsc_BC_var_list,
    qbc_BC_data_bought, qbc_BC_var_list,
    filename="model_improvement_comparison_BC.pdf"
)
plot_model_improvement_comparison(
    vb_SC_data_bought, vb_SC_var_list,
    rsc_SC_data_bought, rsc_SC_var_list,
    qbc_SC_data_bought, qbc_SC_var_list,
    filename="model_improvement_comparison_SC.pdf"
)
# Plotting Variance Reduction Comparisons
plot_variance_reduction_comparison(
    vb_BC_data_bought, vb_BC_var_list, rsc_BC_data_bought, rsc_BC_var_list, qbc_BC_data_bought, qbc_BC_var_list,
    filename='variance_reduction_comparison_BC.pdf'
)

plot_variance_reduction_comparison(
    vb_SC_data_bought, vb_SC_var_list, rsc_SC_data_bought, rsc_SC_var_list, qbc_SC_data_bought, qbc_SC_var_list,
    filename='variance_reduction_comparison_SC.pdf'
)

# Plotting Budget Utilization Comparisons
plot_budget_utilization_comparison(
    vb_BC_data_bought, vb_BC_budget_list, rsc_BC_data_bought, rsc_BC_budget_list, qbc_BC_data_bought, qbc_BC_budget_list,
    filename='budget_utilization_comparison_BC.pdf'
)

plot_budget_utilization_comparison(
    vb_SC_data_bought, vb_SC_budget_list, rsc_SC_data_bought, rsc_SC_budget_list, qbc_SC_data_bought, qbc_SC_budget_list,
    filename='budget_utilization_comparison_SC.pdf'
)



# ======================== Generates Figure 5 in the paper ========================
def plot_percentage_variance_reduction_comparison(
    vb_price_list, vb_variance_list,  
    rsc_price_list, rsc_variance_list,
    qbc_price_list, qbc_variance_list,
    filename
):
    """
    Plot smoothed cumulative mean of (Δv/Δc) vs purchase number.
    y-axis is scaled so that units are [×10^{-3} TWD^2/£].

    For each strategy:
      - Δv_k = v_{k-1} - v_k   (variance drop from buying point k)
      - Δc_k = price_k         (price paid for that point)
      - eff_k = Δv_k / Δc_k
      - y[k] = (eff_1 + ... + eff_k)/k   <-- running average for smoothing
    """

    fig, ax = plt.subplots(figsize=(14, 12))

    # scale the variance traces by 1e3 so values are nicer (1,2,3 instead of 0.001,0.002,...)
    strategies = [
        ("VBAL", vb_price_list,  np.array(vb_variance_list)  * 1e3),
        ("QBCAL",qbc_price_list, np.array(qbc_variance_list) * 1e3),
        ("RSC",  rsc_price_list, np.array(rsc_variance_list) * 1e3),

    ]

    colors  = ['#d62728', '#2ca02c', '#1f77b4']
    markers = ['o',        's',        '^'       ]

    for (strategy_name, price_list, variance_scaled), color, marker in zip(strategies, colors, markers):
        # Δv_k for k >= 1
        # variance_scaled[0] is initial variance
        # variance_scaled[1] is after buying first point, etc.
        delta_v = [
            variance_scaled[i - 1] - variance_scaled[i]
            for i in range(1, len(variance_scaled))
        ]

        # Δc_k: price paid for each purchased point
        # assume first price corresponds to first purchase, etc.
        delta_c = price_list[:len(delta_v)]

        # pointwise efficiency
        point_eff = [
            (dv / dc) if dc != 0 else 0
            for dv, dc in zip(delta_v, delta_c)
        ]

        # running average efficiency:
        # avg_eff[k] = mean(point_eff[0:k+1])
        avg_eff = []
        running_sum = 0.0
        for idx, eff in enumerate(point_eff):
            running_sum += eff
            avg_eff.append(running_sum / (idx + 1))

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
    plt.xlabel('Data points purchased', fontsize=35)
    plt.ylabel(r'$\Delta v / \Delta c$ [×$10^{-3}$ TWD$^2$/£]', fontsize=35)
    max_purchase = max(
        len(vb_variance_list),
        len(rsc_variance_list),
        len(qbc_variance_list),
    )
    step = 2
    xticks = list(range(1, max_purchase + 1, step))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_purchase + 0.5)
    plt.xticks(xticks, fontsize=30)
    ax = plt.gca()
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    # y ticks prettier: show 0, 0.5, 1.2 rather than 0.500000
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}".rstrip('0').rstrip('.')))
    plt.legend(fontsize=30, loc='best')
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


# ======================== Generates Figure 6 in the paper ========================
def plot_sorted_price_bar_chart_with_sc(
    bc_price_list, sc_price_list,
    bc_sellers, sc_sellers,
    strategy_name,
    filename,
    global_min_y,
    global_max_y
):
    # Match sellers between BC and SC
    common_sellers = set(bc_sellers).intersection(sc_sellers)
    if not common_sellers:
        raise ValueError("No common sellers found between BC and SC lists.")

    bc_filtered = [
        (seller, price)
        for seller, price in zip(bc_sellers, bc_price_list)
        if seller in common_sellers
    ]
    sc_filtered = [
        (seller, price)
        for seller, price in zip(sc_sellers, sc_price_list)
        if seller in common_sellers
    ]
    # Sort BC prices in descending order
    bc_filtered.sort(key=lambda x: x[1], reverse=True)
    aligned_bc_prices = [price for _, price in bc_filtered]
    aligned_sc_prices = [dict(sc_filtered)[seller] for seller, _ in bc_filtered]
    ranks = np.arange(1, len(aligned_bc_prices) + 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    # Bars
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
        step = 1
        xticks = list(range(1, num_sellers + 1, step))
        if xticks[-1] != num_sellers:
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
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
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


# ======================== Generates Table 2 in the paper (Wilcoxon signed-rank test) ========================

# --- Helpers ---
def initial_variance_mean_diag(X, Y):
    var_mat, _ = calculate_variance_of_coefficients_on_training(X, Y)
    return float(np.mean(np.diag(var_mat)))

def avg_cost(prices):
    return float(np.mean(prices)) if len(prices) > 0 else np.nan

def run_one_split(seed, price_model, phi=None, B=None, eta_scale=0.5):
    """
    Runs one active learning split for a given random seed and price model.
    Uses independent seeds for data partition, QBC bootstrap, and RSC sampling.
    """
    # defaults if not provided
    if phi is None:
        phi = 1200.0   # default WTP
    if B is None:
        B = 15.0       # default budget

    # --- independent split ---
    X_lab, X_unlab, y_lab, y_unlab = train_test_split(
        x, y, test_size=0.7, shuffle=True, random_state=seed
    )
    X_lab = X_lab.to_numpy(); X_unlab = X_unlab.to_numpy()
    y_lab = y_lab.to_numpy(); y_unlab = y_unlab.to_numpy()

    # --- local parameters ---
    alpha_local = initial_variance_mean_diag(X_lab, y_lab) * 0.8
    eta_local = generate_eta_j(X_unlab, feature_columns, offset=0.1, coeff= eta_scale)

    # --- run three strategies ---
    vb_db, _, _, vb_prices, _ = vb_active_learning(
        X_lab.copy(), y_lab.copy(), X_unlab.copy(), y_unlab.copy(),
        phi, B, alpha_local, eta_local.copy(), price_model=price_model
    )

    bootstrap_seed = seed + 1000
    q_db, _, _, q_prices, _ = qbc_active_learning(
        X_lab.copy(), y_lab.copy(), X_unlab.copy(), y_unlab.copy(),
        phi, B, alpha_local, eta_local.copy(), price_model=price_model,
        n_bootstrap=n_bootstrap, sampling_seed=bootstrap_seed
    )

    rsc_seed = seed + 2000
    r_db, _, _, r_prices, _ = random_sampling_corrected_strategy(
        X_lab.copy(), y_lab.copy(), X_unlab.copy(), y_unlab.copy(),
        phi, eta_local.copy(), B, alpha_local, price_model=price_model, rsc_seed=rsc_seed
    )

    vb_last = vb_db[-1] if vb_db else 0
    q_last  = q_db[-1] if q_db else 0
    r_last  = r_db[-1] if r_db else 0
    return vb_last, q_last, r_last, np.mean(vb_prices), np.mean(q_prices), np.mean(r_prices)

# --- Monte-Carlo runs ---
R = 50
seeds = list(range(R))

avgcost_BC_VBAL, avgcost_BC_QBCAL, avgcost_BC_RSC = [], [], []
avgcost_SC_VBAL, avgcost_SC_QBCAL, avgcost_SC_RSC = [], [], []

# Create output directory for Wilcoxon results
wilcoxon_dir = THIS_DIR / "Plots" / "VarianceScenario"
wilcoxon_dir.mkdir(parents=True, exist_ok=True)
wilcoxon_file = wilcoxon_dir / "Wilcoxon_results.txt"

with open(wilcoxon_file, "w") as f:

    f.write("\n===============================\n")
    f.write("Paired Wilcoxon Test Results\n")
    f.write("===============================\n\n")

    def paired_wilcoxon(vec_a, vec_b, label_a, label_b, scheme):
        a = np.asarray(vec_a)
        b = np.asarray(vec_b)
        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]
        b = b[mask]
        if len(a) == 0:
            line = f"{scheme}: Not enough valid runs for {label_a} vs {label_b}\n"
            f.write(line)
            print(line)
            return
        
        stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
        median_diff = float(np.median(a - b))

        line = (
            f"{scheme}: {label_a} vs {label_b}\n"
            f"   W={stat}, p-value={p:.4g}, median Δ={median_diff:.3f} (n={len(a)})\n\n"
        )

        f.write(line)
        print(line)

    # Run tests and save results
    paired_wilcoxon(avgcost_BC_VBAL,  avgcost_BC_RSC, 'VBAL', 'RSC', 'BC')
    paired_wilcoxon(avgcost_BC_QBCAL, avgcost_BC_RSC, 'QBCAL','RSC', 'BC')
    paired_wilcoxon(avgcost_SC_VBAL,  avgcost_SC_RSC, 'VBAL', 'RSC', 'SC')
    paired_wilcoxon(avgcost_SC_QBCAL, avgcost_SC_RSC, 'QBCAL','RSC', 'SC')

print("\nWilcoxon results saved to:", wilcoxon_file)
print("===============================\nDone ✅")


# ============================================================ Generate Figure 9  in the paper ============================================================
phi_fixed = 1200
B_fixed = B
coeff_values = [0.5, 1, 1.5, 2, 2.5]
B_values_BC = [10, 15, 20, 25, 30] 
B_values_SC = [10, 15, 20, 25, 30]
phi_values = [1000, 1050, 1100, 1150, 1200]

# ======================== Generates Figures 9a and 9b in the paper (sensitivity to WTP φ) ========================
def purchases_vs_phi(phi_list, B_fixed, price_model="BC", random_state=42):
    X_lab, X_unlab, Y_lab, Y_unlab = train_test_split(
        x, y, test_size=0.7, shuffle=True, random_state=random_state
    )

    X_lab = X_lab.to_numpy()
    X_unlab = X_unlab.to_numpy()
    Y_lab = Y_lab.to_numpy()
    Y_unlab = Y_unlab.to_numpy()

    alpha_local = initial_variance_mean_diag(X_lab, Y_lab) * 0.8
    eta_local = generate_eta_j(X_unlab, feature_columns, offset=0.1, coeff=0.5)

    vbal_counts, qbcal_counts, rsc_counts = [], [], []

    for i, phi_val in enumerate(phi_list):
        vb_db, _, _, _, _ = vb_active_learning(
            X_lab.copy(), Y_lab.copy(),
            X_unlab.copy(), Y_unlab.copy(),
            phi_val, B_fixed, alpha_local, eta_local.copy(),
            price_model=price_model
        )

        q_db, _, _, _, _ = qbc_active_learning(
            X_lab.copy(), Y_lab.copy(),
            X_unlab.copy(), Y_unlab.copy(),
            phi_val, B_fixed, alpha_local, eta_local.copy(),
            price_model=price_model,
            n_bootstrap=n_bootstrap,
            sampling_seed=sampling_seed
        )

        r_db, _, _, _, _ = random_sampling_corrected_strategy(
            X_lab.copy(), Y_lab.copy(),
            X_unlab.copy(), Y_unlab.copy(),
            phi_val, eta_local.copy(), B_fixed, alpha_local,
            price_model=price_model,
            rsc_seed = rsc_seed+i

        )

        vbal_counts.append(vb_db[-1] if len(vb_db) else 0)
        qbcal_counts.append(q_db[-1] if len(q_db) else 0)
        rsc_counts.append(r_db[-1] if len(r_db) else 0)

    return np.array(vbal_counts), np.array(qbcal_counts), np.array(rsc_counts)


def plot_sensitivity_to_phi(phi_list, B_fixed, price_model="BC",
                            filename="sensitivity_to_phi.pdf"):
    vbal_counts, qbcal_counts, rsc_counts = purchases_vs_phi(
        phi_list, B_fixed, price_model=price_model
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(phi_list, vbal_counts,
            marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    ax.plot(phi_list, qbcal_counts,
             marker='s', label='QBCAL', color= '#2ca02c', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')
    ax.plot(phi_list, rsc_counts,
             marker='^', label='RSC', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')


    ax.set_xlabel(r'Willingness to Pay $\phi$ [£/TWD$^2$ ]', fontsize=35)
    ax.set_ylabel('Data points purchased', fontsize=35)

    ax.set_xticks(phi_list)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        fontsize=30,
        loc='best'
    )

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()

plot_sensitivity_to_phi(phi_values , B_fixed, price_model="BC",
                        filename="sensitivity_to_phi_BC.pdf")
plot_sensitivity_to_phi(phi_values , B_fixed, price_model="SC",
                        filename="sensitivity_to_phi_SC.pdf")



# ======================== Generates Figures 9c and 9d in the paper (sensitivity to WTS scaling coefficient d1) ========================
def purchases_vs_wts_coeff(coeff_list, phi_fixed, B_fixed,
                           price_model="BC", random_state=42):
    X_lab, X_unlab, Y_lab, Y_unlab = train_test_split(
        x, y, test_size=0.7, shuffle=True, random_state=random_state
    )

    X_lab = X_lab.to_numpy()
    X_unlab = X_unlab.to_numpy()
    Y_lab = Y_lab.to_numpy()
    Y_unlab = Y_unlab.to_numpy()

    alpha_local = initial_variance_mean_diag(X_lab, Y_lab) * 0.8

    vbal_counts, qbcal_counts, rsc_counts = [], [], []

    for i, coeff in enumerate(coeff_list):
        eta_local = generate_eta_j(X_unlab, feature_columns, offset=0.1, coeff=coeff)

        vb_db, _, _, _, _ = vb_active_learning(
            X_lab.copy(), Y_lab.copy(),
            X_unlab.copy(), Y_unlab.copy(),
            phi_fixed, B_fixed, alpha_local, eta_local.copy(),
            price_model=price_model
        )

        q_db, _, _, _, _ = qbc_active_learning(
            X_lab.copy(), Y_lab.copy(),
            X_unlab.copy(), Y_unlab.copy(),
            phi_fixed, B_fixed, alpha_local, eta_local.copy(),
            price_model=price_model,
            n_bootstrap=n_bootstrap,
            sampling_seed=sampling_seed
        )

        r_db, _, _, _, _ = random_sampling_corrected_strategy(
            X_lab.copy(), Y_lab.copy(),
            X_unlab.copy(), Y_unlab.copy(),
            phi_fixed, eta_local.copy(), B_fixed, alpha_local,
            price_model=price_model,
            rsc_seed = rsc_seed+i
        )

        vbal_counts.append(vb_db[-1] if len(vb_db) else 0)
        qbcal_counts.append(q_db[-1] if len(q_db) else 0)
        rsc_counts.append(r_db[-1] if len(r_db) else 0)

    return np.array(vbal_counts), np.array(qbcal_counts), np.array(rsc_counts)


def plot_sensitivity_to_wts_coeff(coeff_list, phi_fixed, B_fixed,
                                  price_model="BC",
                                  filename="sensitivity_to_wts_coeff.pdf"):
    vbal_counts, qbcal_counts, rsc_counts = purchases_vs_wts_coeff(
        coeff_list, phi_fixed, B_fixed, price_model=price_model
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(coeff_list, vbal_counts,
            marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    ax.plot(coeff_list, qbcal_counts,
            marker='s', label='QBCAL', color= '#2ca02c', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')
    ax.plot(coeff_list, rsc_counts,
            marker='^', label='RSC', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')

    ax.set_xlabel(r'WTS scaling coefficient $d_1$', fontsize=35)
    ax.set_ylabel('Data points purchased', fontsize=35)

    ax.set_xticks(coeff_list)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)

    ax.legend(
        fontsize=30,
        loc='lower left'
    )

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    _add_axis_arrows(ax, lw=2.5)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()



plot_sensitivity_to_wts_coeff(
    coeff_values, phi_fixed, B_fixed,
    price_model="BC",
    filename="sensitivity_to_wts_coeff_BC.pdf"
)

plot_sensitivity_to_wts_coeff(
    coeff_values, phi_fixed, B_fixed,
    price_model="SC",
    filename="sensitivity_to_wts_coeff_SC.pdf"
)

# ======================== Generates Figures 9e and 9f in the paper (sensitivity to budget limit B) ========================
def purchases_vs_budget(B_list, price_model="BC", random_state=42):
    X_lab, X_unlab, y_lab, y_unlab = train_test_split(
        x, y, test_size=0.7, shuffle=True, random_state=random_state
    )

    X_lab = X_lab.to_numpy()
    X_unlab = X_unlab.to_numpy()
    y_lab = y_lab.to_numpy()
    y_unlab = y_unlab.to_numpy()

    # very loose target so they keep buying
    alpha_local = initial_variance_mean_diag(X_lab, y_lab) * 0.1

    eta_local = generate_eta_j(X_unlab, feature_columns, offset=0.1, coeff=0.5)

    vbal_counts, qbcal_counts, rsc_counts = [], [], []

    for i, B_val in enumerate(B_list):
        vb_db, _, _, _, _ = vb_active_learning(
            X_lab.copy(), y_lab.copy(),
            X_unlab.copy(), y_unlab.copy(),
            phi, B_val, alpha_local, eta_local.copy(),
            price_model=price_model
        )

        q_db, _, _, _, _ = qbc_active_learning(
            X_lab.copy(), y_lab.copy(),
            X_unlab.copy(), y_unlab.copy(),
            phi, B_val, alpha_local, eta_local.copy(),
            price_model=price_model,
            n_bootstrap=n_bootstrap,
            sampling_seed=sampling_seed
        )

        r_db, _, _, _, _ = random_sampling_corrected_strategy(
            X_lab.copy(), y_lab.copy(),
            X_unlab.copy(), y_unlab.copy(),
            phi, eta_local.copy(), B_val, alpha_local,
            price_model=price_model,
            rsc_seed = rsc_seed+i

        )

        vbal_counts.append(vb_db[-1])
        qbcal_counts.append(q_db[-1])
        rsc_counts.append(r_db[-1])

    return np.array(vbal_counts), np.array(qbcal_counts), np.array(rsc_counts)


def plot_sensitivity_to_budget(B_list, price_model="BC", filename="sensitivity_to_budget.pdf"):
    vbal_counts, qbcal_counts, rsc_counts = purchases_vs_budget(
        B_list, price_model=price_model
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(B_list, vbal_counts,
            marker='o', label='VBAL', color='#d62728', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor='#d62728',markeredgecolor='white')
    ax.plot(B_list, qbcal_counts,
            marker='s', label='QBCAL', color= '#2ca02c', 
            linewidth=3, markersize=25, markeredgewidth=1.2, markerfacecolor= '#2ca02c', markeredgecolor='white')

    ax.plot(B_list, rsc_counts,
             marker='^', label='RSC', color='#1f77b4', 
            linewidth=3,  markersize=25, markeredgewidth=1.2, markerfacecolor='#1f77b4', markeredgecolor='white')

    ax.set_xlabel('Budget limit $B$ [£]', fontsize=35)
    ax.set_ylabel('Data points purchased', fontsize=35)

    ax.set_xticks(B_list)
    ax.tick_params(axis='x', width=2.5, length=8, direction='out', labelsize=30)
    ax.tick_params(axis='y', width=2.5, length=8, direction='out', labelsize=30)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        fontsize=30,
        loc='upper left'
    )

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    _add_axis_arrows(ax, lw=2.5)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    save_fig_consistent(fig, filename, width=14, height=12, dpi=300)
    plt.show()
plot_sensitivity_to_budget(B_values_BC, price_model="BC", filename="sensitivity_to_budget_BC.pdf") 
plot_sensitivity_to_budget(B_values_SC, price_model="SC", filename="sensitivity_to_budget_SC.pdf")


# ============================================================
# Appendix: Generate Table 4 and 5 for Monte Carlo robustness evaluation (50 independent partitions)
# ============================================================


def monte_carlo_summary(param_list, B_fixed, price_model="BC", runs=50, param_name="phi"):
    """
    Perform Monte Carlo evaluation across independent data partitions.
    Each run uses:
        - data_seed = 100 + s  (s ∈ [0, runs))
        - QBC bootstrap_seed = 1100 + s
        - RSC sampling_seed   = 2100 + s
    """
    records = []
    for val in param_list:
        vb_counts, q_counts, r_counts = [], [], []
        for s in range(runs):
            vb_c, q_c, r_c, _, _, _ = run_one_split(
                seed=100 + s,
                price_model=price_model,
                phi=val if param_name == "phi" else 1200,
                B=val if param_name == "B" else B_fixed,
                eta_scale=val if param_name == "eta" else 0.5
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


# === Run and print the Monte Carlo summary ===


# Buyer-Centric
df_wtp_bc = monte_carlo_summary(phi_values, B_fixed, price_model="BC", runs=50, param_name="phi")
df_wts_bc = monte_carlo_summary(coeff_values, B_fixed, price_model="BC", runs=50, param_name="eta")
df_b_bc   = monte_carlo_summary(B_values_BC, B_fixed=None, price_model="BC", runs=50, param_name="B")

# Seller-Centric
df_wtp_sc = monte_carlo_summary(phi_values, B_fixed, price_model="SC", runs=50, param_name="phi")
df_wts_sc = monte_carlo_summary(coeff_values, B_fixed, price_model="SC", runs=50, param_name="eta")
df_b_sc   = monte_carlo_summary(B_values_SC, B_fixed=None, price_model="SC", runs=50, param_name="B")

# Print
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
# === Save sensitivity analysis results ===
sensitivity_results = {
    "WTP_BC": df_wtp_bc,
    "WTP_SC": df_wtp_sc,
    "WTS_BC": df_wts_bc,
    "WTS_SC": df_wts_sc,
    "Budget_BC": df_b_bc,
    "Budget_SC": df_b_sc,
}

# Create output directory
output_dir = THIS_DIR / "Plots" / "Sensitivity_Results"
output_dir.mkdir(parents=True, exist_ok=True)

# Save each table as CSV
for name, df in sensitivity_results.items():
    df.to_csv(output_dir / f"{name}.csv", index=False)

print("\nSensitivity analysis tables saved to:")
print(output_dir)

