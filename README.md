# **`README.md`**

# Adaptive Macro-Financial Risk Calibration for Protectionist Environments

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.25285v1-b31b1b.svg)](https://arxiv.org/abs/2603.25285v1)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2603.25285v1)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20Econometrics%20%7C%20Risk%20Management-00529B)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)
[![Data Sources](https://img.shields.io/badge/Data-Yahoo%20Finance%20%7C%20Caldara%20et%20al.%20(2020)-lightgrey)](https://finance.yahoo.com/)
[![Core Method](https://img.shields.io/badge/Method-GJR--DCC--X-orange)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)
[![Analysis](https://img.shields.io/badge/Analysis-Structural%20Breaks%20%7C%20IRF-red)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)
[![Validation](https://img.shields.io/badge/Validation-Model%20Confidence%20Set%20(MCS)-green)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)
[![Robustness](https://img.shields.io/badge/Robustness-10Y%20T--Bond%20%7C%20VIX%20%7C%20EPU-yellow)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Numba](https://img.shields.io/badge/Numba-%2300A3E0.svg?style=flat&logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships)

**Repository:** `https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Shifting Correlations: How Trade Policy Uncertainty Alters stock-T bill Relationships"** by:

*   **Demetrio Lacava** (Department of Economics, University of Messina)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, highly optimized pipeline that executes the entire research workflow: from the rigorous ingestion and stationarity transformation of high-frequency market data to the execution of a Numba-compiled GJR-DCC-X econometric engine, culminating in out-of-sample portfolio optimization and Model Confidence Set (MCS) evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_full_research_replication`](#key-callable-execute_full_research_replication)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Lacava (2026). The core of this repository is the iPython Notebook `how_trade_policy_uncertainty_alters_stock_tbill_relationships_draft.ipynb`, which contains a comprehensive suite of 39+ orchestrated tasks to replicate the paper's findings.

The pipeline addresses a critical vulnerability in modern portfolio theory: the assumption that short-term government bonds act as a reliable safe haven during equity market drawdowns. The author demonstrates that exogenous policy shocks—specifically Trade Policy Uncertainty (TPU)—can fundamentally alter the correlation-generating process, driving stock-bond correlations positive and triggering "liquidation cascades."

The codebase operationalizes the proposed solution:
-   **Filters** univariate asset returns for asymmetric volatility clustering using the GJR-GARCH(1,1) model.
-   **Augments** the Dynamic Conditional Correlation (DCC) framework to directly ingest exogenous policy vectors ($x_{t-1}$) and political regime indicators ($D_t$).
-   **Evaluates** the economic utility of the augmented models via out-of-sample Global Minimum Variance (GMV) portfolio optimization.
-   **Validates** the statistical superiority of the policy-aware models using the Hansen-Lunde-Nason (2011) Model Confidence Set procedure.

## Theoretical Background

The implemented methods combine techniques from Time-Series Econometrics, Multivariate Volatility Modeling, and Mathematical Finance.

**1. Univariate Volatility Filtering (GJR-GARCH):**
To isolate the pure correlation signal, the marginal distribution of each asset is filtered for conditional heteroskedasticity and the "leverage effect" (asymmetric response to negative shocks):
$$ h^2_{i,t} = \omega_i + \alpha_i r^2_{i,t-1} + \beta_i h^2_{i,t-1} + \gamma_i r^2_{i,t-1} I_{i,t-1} $$

**2. Augmented Correlation Dynamics (DCC-X):**
The core innovation integrates the exogenous Trade Policy Uncertainty index ($x_{t-1}$) directly into the quasi-correlation evolution, transitioning from an endogenous-only news tracker to a macro-aware risk calibrator:
$$ Q_t = (1-\theta_1-\theta_2-\theta_3 \bar{x})\bar{Q} + \theta_1 \tilde{\epsilon}_{t-1}\tilde{\epsilon}'_{t-1} + \theta_2 Q_{t-1} + \theta_3 x_{t-1}\bar{Q} $$

**3. Structural Break Testing:**
The pipeline tests for discrete shifts in the sensitivity of correlations to trade shocks following major tariff announcements (e.g., March 2018), utilizing a contemporaneous step dummy ($D_t$):
$$ Q_t = (1-\theta_1-\theta_2-(\theta_3+\delta D_t)\bar{x})\bar{Q} + \dots + (\theta_3+\delta D_t)x_{t-1}\bar{Q} $$

**4. Economic Loss Evaluation (GMV):**
The out-of-sample utility of the forecasted covariance matrices ($\hat{H}_\tau$) is evaluated by constructing the Global Minimum Variance portfolio:
$$ \hat{v}_\tau = \sqrt{n} \frac{\hat{H}^{-1}_\tau \mathbf{j}_n}{\mathbf{j}'_n \hat{H}^{-1}_\tau \mathbf{j}_n} $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships/blob/main/how_trade_policy_uncertainty_alters_stock_tbill_relationships_ipo_main.png" alt="Adaptive Macro-Financial Risk Calibration Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`how_trade_policy_uncertainty_alters_stock_tbill_relationships_draft.ipynb`) implements the full research pipeline, including:

-   **Numba JIT-Compiled Recursions:** The computationally intensive GJR-GARCH and DCC-X matrix evolutions are compiled to C-level machine code, reducing optimization times from hours to seconds.
-   **Mathematically Exact Inference:** Implements the true White (1980) heteroskedasticity-consistent covariance matrix (the sandwich estimator: $\hat{A}^{-1}\hat{B}\hat{A}^{-1}$) via second-order numerical Hessians, discarding flawed OPG approximations.
-   **Rigorous Matrix Regularization:** Replaces naive diagonal jitter with mathematically rigorous Eigenvalue Clipping to project non-Positive Definite (PD) matrices onto the space of valid correlation matrices.
-   **Zero-Leakage State Machine:** The out-of-sample forecasting engine utilizes a strict State Machine architecture, explicitly passing terminal in-sample states and updating them sequentially with realized data to absolutely guarantee zero look-ahead bias.
-   **Scale-Aware Optimization:** The Sequential Least SQuares Programming (SLSQP) optimizer is initialized with dynamic, scale-aware starting points to navigate severe multicollinearity (e.g., TPU vs. Interaction term) and vastly different regressor domains (e.g., VIX vs. Binary Dummies).
-   **Configuration-Driven Design:** All study parameters, temporal boundaries, and optimization constraints are managed in an external `config.yaml` file, ensuring strict methodological reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Engineering (Tasks 1-13):** Ingests raw equity prices, Treasury yields, and policy indices. Enforces strict chronological alignment via index intersection, computes stationary log-returns and basis-point changes, and engineers lagged regressors and interaction terms.
2.  **Univariate Filtering (Tasks 14-16):** Estimates the GJR-GARCH(1,1) parameters for all assets via constrained MLE, extracts conditional variances, and computes unit-variance standardized residuals ($\tilde{\epsilon}_{i,t}$).
3.  **Multivariate Estimation (Tasks 17-21):** Estimates the baseline DCC and augmented DCC-X models. Computes robust standard errors, AIC/BIC, Likelihood Ratio tests, and Ljung-Box residual diagnostics.
4.  **Visualization & IRF (Tasks 22-26):** Renders the residual correlation heatmap, 60-day rolling correlations, and conditional correlation overlays. Simulates the Impulse Response Function (IRF) to a 1-std-dev TPU shock and benchmarks it against a VIX shock.
5.  **Out-of-Sample Forecasting (Tasks 27-31):** Partitions the sample, generates 583 one-step-ahead covariance forecasts, computes statistical (Frobenius, QLike) and economic (GMV, RPV) losses, and executes the stationary bootstrap Model Confidence Set (MCS) procedure.
6.  **Structural Analysis & Robustness (Tasks 32-39):** Estimates structural break models around key tariff dates, computes the 750-day rolling $\hat{\theta}_3$ path, and executes robustness checks replacing the 3M T-bill with the 10Y T-bond and controlling for VIX/EPU.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 39 major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `execute_full_research_replication`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_full_research_replication`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data validation, univariate filtering, multivariate optimization, structural break testing, out-of-sample forecasting, and the final cryptographic fidelity audit.

## Prerequisites

-   Python 3.10+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `numba`, `matplotlib`, `seaborn`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships.git
    cd how_trade_policy_uncertainty_alters_stock_tbill_relationships
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy numba matplotlib seaborn pyyaml
    ```

## Input Data Structure

The pipeline requires three primary data structures, strictly validated at runtime:

1.  **`df_assets_raw` (pd.DataFrame):** Contains the raw price levels of the four equity indices (`SP500_Price`, `DJIA_Price`, `NASDAQ_Price`, `RUSSELL_Price`) and the raw annualized yields (`T_Bill_Yield`, `T_Bond_10Y_Yield`). Indexed by observed trading dates.
2.  **`df_exogenous_raw` (pd.DataFrame):** Contains the observable policy factors (`TPU_Index`, `VIX_Index`, `EPU_Index`).
3.  **`df_metadata_raw` (pd.DataFrame):** Contains the categorical logic gates (`Pres_Dummy`, `D_Mar2018`, `D_May2019`, `D_Apr2025`) used for interaction terms and structural break analysis.

*Note: The pipeline includes a high-fidelity synthetic data generator for testing purposes if access to the original Yahoo Finance or Iacoviello TPU data is unavailable.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic data, and use the top-level orchestrator to execute the baseline pipeline and all robustness checks:

```python
import os
import yaml
import pandas as pd
import numpy as np

# 1. Load the master configuration from the YAML file.
# (Assumes config.yaml is in the working directory)
with open("config.yaml", "r", encoding="utf-8") as f:
    study_config = yaml.safe_load(f)

# 2. Load raw datasets (Example using synthetic generator provided in the notebook)
# In production, load from CSV: pd.read_csv("data/assets_raw.csv")
df_assets_raw, df_exogenous_raw, df_metadata_raw = generate_synthetic_research_data()

# 3. Execute the entire replication study.
master_results = execute_full_research_replication(
    df_assets_raw=df_assets_raw,
    df_exogenous_raw=df_exogenous_raw,
    df_metadata_raw=df_metadata_raw,
    config=study_config
)

# 4. Access results
if master_results:
    baseline_bundle = master_results["Baseline_ArtifactBundle"]
    
    print("\n[RESULTS] Table 2b: DCC-X Parameter Estimates")
    print(baseline_bundle.dcc_params_table)
    
    print("\n[RESULTS] Table 3: Model Confidence Set (MCS) P-Values")
    if baseline_bundle.mcs_pvalue_table is not None:
        print(baseline_bundle.mcs_pvalue_table)
    
    print("\n[AUDIT] Final Fidelity Audit Report")
    print(master_results["Fidelity_Audit"]["DiagnosticReportString"])
```

## Output Structure

The pipeline returns a master dictionary containing:
-   **`Baseline_ArtifactBundle`**: The core pipeline artifacts, including Tables 1, 2a, 2b, 2c, 3, 4, and Figures 1-5.
-   **`Robustness_10Y_Bond`**: The artifacts and audit report for the maturity robustness check (Table 5a).
-   **`Robustness_VIX_EPU`**: The artifacts and audit report for the incremental information check (Table 5b).
-   **`Fidelity_Audit`**: The comprehensive parameter-by-parameter comparison against the manuscript's published targets, including root-cause diagnostics for any deviations.
-   **`Final_Replication_Package`**: The compiled dictionary of exhibits, the methodological conventions documentation, and the SHA-256 cryptographic integrity manifest.

## Project Structure

```
how_trade_policy_uncertainty_alters_stock_tbill_relationships/
│
├── how_trade_policy_uncertainty_alters_stock_tbill_relationships_draft.ipynb   # Main implementation notebook
├── config.yaml                                                                 # Master configuration file
├── requirements.txt                                                            # Python package dependencies
│
├── LICENSE                                                                     # MIT Project License File
└── README.md                                                                   # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Optimization Constraints:** Adjust the lower bounds for $\omega$ or the strictness of the stationarity inequality constraints.
-   **Econometrics:** Alter the rolling window size ($W=750$), the IRF simulation horizon ($H=120$), or the specific lags applied to the Ljung-Box diagnostics.
-   **Out-of-Sample Settings:** Modify the temporal split boundaries, the ridge regularization penalty ($\lambda$), or the MCS bootstrap replications ($B=5000$).

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Correlation Topologies:** Replacing the scalar DCC approach with the **DCC-MIDAS** (Colacito et al., 2011) framework to handle mixed-frequency regressors natively, or **Smooth Transition Correlation** models (Silvennoinen and Teräsvirta, 2015) to endogenize the regime shifts.
-   **Expanded Asset Universes:** Adapting the pipeline to evaluate the safe-haven status of Gold, Swiss Francs, or Cryptocurrencies during trade shocks.
-   **Dynamic Portfolio Constraints:** Integrating transaction costs, turnover penalties, or long-only constraints into the GMV optimization routine to simulate real-world liquidation cascades more accurately.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{lacava2026shifting,
  title={Shifting Correlations: How Trade Policy Uncertainty Alters stock-T bill Relationships},
  author={Lacava, Demetrio},
  journal={arXiv preprint arXiv:2603.25285v1},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Adaptive Macro-Financial Risk Calibration for Protectionist Environments: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/how_trade_policy_uncertainty_alters_stock_tbill_relationships
```

## Acknowledgments

-   Credit to **Demetrio Lacava** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **SciPy**, **Numba**, and **Pandas** contributors.

--

*This README was generated based on the structure and content of the `how_trade_policy_uncertainty_alters_stock_tbill_relationships_draft.ipynb` notebook and follows best practices for research software documentation.*
