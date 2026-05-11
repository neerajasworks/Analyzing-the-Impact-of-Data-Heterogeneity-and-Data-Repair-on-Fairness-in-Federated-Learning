**Analyzing the Impact of Data Heterogeneity and DataRepair on Fairness in Federated Learning**

This repository contains the full experimental code used to produce the results of the paper "Analyzing the Impact of Data Heterogeneity and DataRepair on Fairness in Federated Learning". 
The experiments study how different data distribution strategies across federated clients affect the fairness of a trained binary classifier, and whether a pre-processing repair technique (Random Repair via Optimal Transport) can mitigate observed bias.


**Table of Contents**

1. Project Overview
2. Repository Structure
3. Requirements and Installation
4. Dataset Setup
5. Code Architecture
   * Functions.py
   * Main_code.ipynb
6. Key Configuration Parameters
7. Experiments
8. Fairness Metrics
9. Output Files
10. Reproducing Results
11. Notes and Troubleshooting

**Project Overview**
The code trains a binary income classifier (predicting whether income exceeds $50K/year) on the UCI Adult dataset using both centralised and federated learning (via TensorFlow Federated). The core research questions are:

* How does non-i.i.d. data partitioning (heterogeneity) across federated clients affect fairness with respect to sensitive attributes such as Sex or Race/Ethnicity?
* Does applying Random Repair (an Optimal Transport-based pre-processing technique) before training reduce bias — and at what cost to accuracy?
* How does the client selection strategy (random, majority-biased, minority-biased) interact with fairness outcomes?

Five main experiments are provided, each varying the data distribution mode and client selection strategy. Each experiment is repeated 50 times and results are aggregated with 95% confidence intervals.

**Requirements and Installation**
Python Version
Python 3.8–3.10 is recommended (TensorFlow Federated has strict version constraints).

Core Dependencies
pip install tensorflow==2.14.*
pip install tensorflow-federated==0.86.*
pip install numpy pandas matplotlib scipy scikit-learn POT
