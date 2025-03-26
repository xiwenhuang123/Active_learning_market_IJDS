# Active_learning_market_IJDS

This repository contains the code and data for the paper: "How to Purchase Labels? A Cost-effective Approach Using Active Learning Markets" submitted to *INFORMS Journal on Data Science (IJDS)*.

## Abstract

We introduce and analyse active learning markets as a way to purchase labels, in situations where analysts aim to acquire additional data to improve model fitting, or to better train models for predictive analytics applications. This comes in contrast to the many proposals that already exist to purchase
features and examples. By originally formalizing the market clearing as an opti-
mization problem, we integrate budget constraints and improvement thresholds
into the data selection process, ensuring efficiency and cost-effectiveness. We
focus on a single-buyer-multiple-seller setup and propose the use of two active
learning strategies (variance based and query-by-committee based), paired with
distinct pricing mechanisms. They are compared to a benchmark random sam-
pling approach. The proposed strategies are validated on real-world datasets
from two critical domains: real estate pricing and energy forecasting. Results
demonstrate the robustness of our approach, consistently achieving superior
performance with fewer data points compared to conventional methods. Our
proposal comprises an easy-to-implement practical solution for optimizing data
acquisition in resource-constrained environments
## Development Environment Setup

This project utilizes ```Python 3.11.7```. It's recommended to create a virtual environment to manage dependencies and ensure consistency across different development setups.

### Using Conda

If you have Conda installed, you can create and activate a new environment with the following commands:

```bash
# Create a new conda environment named 'project_env' with Python 3.11.7
conda create -n project_env python=3.11.7

# Activate the newly created environment
conda activate project_env
