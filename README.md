# Active_learning_market_IJDS
Repository for the IJDS submission by **Xiwen Huang**

This repository contains the code and data for the paper:  
**"How to Purchase Labels? A Cost-effective Approach Using Active Learning Markets"**,  
submitted to *INFORMS Journal on Data Science (IJDS)*.

## Abstract

We introduce and analyse active learning markets as a way to purchase labels, in situations where analysts aim to acquire additional data to improve model fitting, or to better train models for predictive analytics applications. This contrasts with many existing proposals that focus on purchasing features and examples. By formalizing the market clearing as an optimization problem, we integrate budget constraints and improvement thresholds into the data selection process, ensuring efficiency and cost-effectiveness.  

We focus on a single-buyer–multiple-seller setup and propose two active learning strategies (variance-based and query-by-committee based), combined with distinct pricing mechanisms. These strategies are compared to a benchmark random sampling approach. The proposed methods are validated on real-world datasets from two domains: real estate pricing and energy forecasting. Results demonstrate strong robustness and consistent performance gains with fewer acquired data points. Our proposal offers a practical and easy-to-implement solution for optimizing data acquisition in resource-constrained environments.

---

## 📦 Data Location (Required for Reproducibility)

> **Important:**  
> All datasets used in the predictive-ability scenario must be placed inside  
> `data/Hog_buildings/`.  
> The repository uses **relative paths only**.  
> No machine-specific absolute paths (e.g., `/Users/...`) are used, ensuring that all scripts run on any machine without modification.

---

## Code Organization

To run the experiments:

---

### **1. Estimation-quality–focused scenario**

- `IJDS_Variance.py`: runs the real-estate data in the estimation-quality–focused scenario (Section 4.1.1–4.1.5).  
  **Reproduces:** Figures 3–6, Figures 9a–9f, Table 2, and Tables 4–5 (Appendix).  
  Also generates additional diagnostic plots not included in the paper.

- `Monte_carlo_variance_scenario.py`: runs the Monte Carlo simulation for the estimation-quality–focused scenario  
  (Section 4.1.5: Data Variability).  
  **Reproduces:** Figures 7–8.

---

### **2. Predictive-ability–focused scenario**

- `IJDS_MSE.py`: runs the energy building data (`Hog_industrial_Rachael.csv` and `Hog_industrial_Madge.csv`)  
  in the predictive-ability–focused scenario (Section 4.2.1–4.2.5).  
  **Reproduces:** Figures 10–11, Figures 14a–14f, and Tables 6–7 (Appendix).  
  Also generates additional diagnostic plots not included in the paper.

- `Monte_carlo_MSE_scenario.py`: runs the Monte Carlo simulation using the energy building data  
  in the predictive-ability–focused scenario (Section 4.2.5: Data Variability).  
  **Reproduces:** Figures 12–13.

---

### Input data and outputs

- `data/`: includes the real estate dataset (`Real estate valuation data set.xlsx`)  
  and building energy datasets in `Hog_buildings/`  
  (select `Hog_industrial_Rachael.csv` and `Hog_industrial_Madge.csv` for reproducibility).

- `Plots/`: contains all experiment output figures for Section 4  
  (including additional diagnostic plots not shown in the paper).

---

## Reproducibility Instructions

To reproduce all experiments:

1. **Clone this repository**
   ```bash
   git clone https://github.com/xiwenhuang123/Active_learning_market_IJDS.git
   cd Active_learning_market_IJDS

2. **Create and activate the environment**
   This project uses **Python 3.11.7**.  
   It is recommended to create a virtual environment to ensure consistent dependencies.
   If using Conda:
   ```bash
   # Create a new conda environment with Python 3.11.7
   conda create -n project_env python=3.11.7
   # Activate the environment
   conda activate project_env
   ```
3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
4. **Run experiments**
   ```
   python IJDS_Variance.py
   python IJDS_MSE.py
   python Monte_carlo_variance_scenario.py
   python Monte_carlo_MSE_scenario.py
   ```

For any inquiries, please contact:
📧 xiwen.huang23@ic.ac.uk
