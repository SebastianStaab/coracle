## Coracle - A Machine Learning Framework to Identify Bacteria Associated with Continous Variables 
# Ensemble Machine Learning for Coral Microbiome Analysis

Open access web server: [micportal.org ](https://micportal.org)
Zenodo: [![DOI](https://zenodo.org/badge/676863744.svg)](https://doi.org/10.5281/zenodo.19050205)
Research Paper: [https://doi.org/10.1093/bioinformatics/btad749 ](https://doi.org/10.1093/bioinformatics/btad749 )
cfs.py is an implementation of the Correlation Based Feature Selection by Mark Hall (https://researchcommons.waikato.ac.nz/handle/10289/1024)

Coracle is an artificial intelligence framework that utilizes an ensemble approach of different feature selection methods and machine learning models to identify bacteria strains associated with continuous physiological variables. It is specifically designed to maximize the utility of small and medium-sized datasets, such as those typically found in coral microbiome research. The framework is optimized for the taxonomic levels of "Family" and "Order," which provide a balanced level of detail and feature count for robust analysis.The core logic of the framework is implemented in the file named coracle.py. Comprehensive implementation details and guidance are provided in the Coracle Research Paper.

# Description of Methodology
The goal of this framework is to consistently select bacterial strains associated with specific traits, such as heat stress resistance, while avoiding the pitfalls of overfitting and high dimensionality.
Normalization: The system applies two distinct normalization methods, including relative abundance ($L_1$-normalization) and centered log ratio (CLR), to the input feature counts.
Ensemble Feature Selection: Within a leave-one-out cross-validation loop, the framework employs three different selection methods: Lasso, Adaptive Lasso (Alasso), and Correlation-based Feature Selection (CFS).
Machine Learning Modeling: Selected features are processed using Random Forest Regression to evaluate their predictive power and importance.
Scoring System: The results are aggregated by a final score that incorporates the performance of the various machine learning models, the respective feature importance, and the frequency with which a feature was selected across different models.

# Input Format
The framework requires two primary input datasets to execute the analysis:
1. Feature data file (x): This is a sample-by-feature matrix where rows represent individual samples and columns represent features, such as ASVs or gene IDs[cite: 1]. This table must contain only numeric data, representing raw counts or relative abundances. Non-numeric metadata columns should be excluded from this file.
2. Target variable data file (y): This is a single-column table containing the continuous target variable, such as pH or temperature, for each sample[cite: 1, 2]. The index of this file must match the sample IDs provided in the feature table to ensure data alignment.

# Format Summary
| File | Orientation | Index | Columns | Notes |
|------|-------------|-------|---------|-------|
| `features.csv` | Samples × Features | Sample IDs | Feature IDs | Numeric only |
| `target.csv`   | Samples × 1         | Sample IDs | Target name | One column |

# Core Parameters
The analysis can be customized using the following parameters found in the coracle function:
alpha_l1: This is the lambda or alpha value for $L_1$-lasso traverses.
alpha_clr: This is the lambda or alpha value for CLR-lasso traverses.
random_state: This is an optional integer seed used to help making the results reproducible and deterministic (not supported for alasso so far).

# Output and Results
The framework produces a comprehensive table that serves as the final result of the analysis.
Scoring and Ranking: Bacterial strains are ranked according to a final score that weights their importance across all ensemble models.
Model Performance: The output includes the performance metrics for each model traverse, specifically the $R^2$ score and Mean Squared Error (MSE).
Coefficients: Regression coefficients from Lasso and Adaptive Lasso are provided to offer insights into the direction and strength of the association for each identified strain.
Feature Importance: Specific importance values from the Random Forest Regression models are included for each feature.
