# Active_learning_market_IJDS

This repository contains the code and data for the paper: "How to Purchase Labels? A Cost-effective Approach Using Active Learning Markets" submitted to *INFORMS Journal on Data Science (IJDS)*.

## Abstract

We introduce and analyse active learning markets as a way to purchase labels, in situations where analysts aim to acquire additional data to improve model fitting, or to better train models for predictive analytics applications. This comes in contrast to the many proposals that already exist to purchase features and examples. By originally formalizing the market clearing as an optimization problem, we integrate budget constraints and improvement thresholds into the data selection process, ensuring efficiency and cost-effectiveness. We focus on a single-buyer-multiple-seller setup and propose the use of two activelearning strategies (variance based and query-by-committee based), paired with distinct pricing mechanisms. They are compared to a benchmark random sampling approach. The proposed strategies are validated on real-world datasets from two critical domains: real estate pricing and energy forecasting. Results demonstrate the robustness of our approach, consistently achieving superior
performance with fewer data points compared to conventional methods. Our proposal comprises an easy-to-implement practical solution for optimizing data acquisition in resource-constrained environments.


## Code Organization

To run the experiments:

-  `Variance_scenario.py` : run the real estate data in the variance-dependent scenario (Section 4.1.1 to 4.1.3).

- `MSE_scenario.py`: run the energy building data "Hog_industrial_Rachael.csv" and "Hog_industrial_Madge.csv" in the MSE-dependent scenario (Section 4.2.1 to 4.2.3).

- `Monte_carlo_variance_scenario.py`: run the monte carlo simulation using the real estate data in the variance-dependent scenario (Section 4.1.5).
- `Monte_carlo_MSE_scenario.py`: run the monte carlo simulation using the energy building data in the MSE-dependent scenario (Section 4.2.5).

Input data and results:

- `\Data`: includes real estate data `Real estate valuation data set.xlsx` and energy building data in the file `Hog_Buidings` (please select "Hog_industrial_Rachael.csv" and "Hog_industrial_Madge.csv" for reproducability).

- `\Plots`: includes all the output in each scenario in Section 4 (includes additional figures not mentioned in this paper).

## Development environment setup

This project utilizes ```Python 3.11.7```. It's recommended to create a virtual environment to manage dependencies and ensure consistency across different development setups.
If you have Conda installed, you can create and activate a new environment with the following commands:
```bash
# Create a new conda environment named 'project_env' with Python 3.11.7
conda create -n project_env python=3.11.7

# Activate the newly created environment
conda activate project_env
```
Then install all the necessary packages using ```pip install -r requirement.txt```.
-For any inquiries, please contact: ```xiwen.huang23@ic.ac.uk```




