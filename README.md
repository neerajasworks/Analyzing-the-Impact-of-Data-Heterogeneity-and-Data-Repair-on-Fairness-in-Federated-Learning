**Analyzing the Impact of Data Heterogeneity and DataRepair on Fairness in Federated Learning**

This repository contains the full experimental code used to produce the results of the paper "Analyzing the Impact of Data Heterogeneity and DataRepair on Fairness in Federated Learning". 
The experiments study how different data distribution strategies across federated clients affect the fairness of a trained binary classifier, and whether a pre-processing repair technique (Random Repair via Optimal Transport) can mitigate observed bias.


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

```python
pip install tensorflow==2.14.*
pip install tensorflow-federated==0.86.*
pip install numpy pandas matplotlib scipy scikit-learn POT
```

**Dataset Setup**

This project uses the UCI Adult (Census Income) dataset.

Download from the UCI ML Repository https://archive.ics.uci.edu/dataset/2/adult

**Code Architecture**

`Functions.py` : This file contains all reusable logic, organised into clearly labelled sections.
`Main_code.ipynb` :  The notebook is the experiment runner. Each cell sets up configuration variables.

**Reproducing Results**

* Clone the repo and install all dependencies (see Requirements).
* Set up the Adult dataset (see Dataset Setup).
* Open Main_code.ipynb in Jupyter:
  
```python
bash   jupyter notebook Main_code.ipynb
```

Run cells in order. Each experiment cell is self-contained. You can run them independently by setting `MODO_REPARTO` and other variables as desired.

Results are printed to stdout (with confidence intervals) and saved as figures to the Figuras/ directory.


**Estimated runtime**: With `N_REPETICIONES=50`, `NUM_CLIENTES=100`, and `N_ROUNDS=64`, each experiment takes several hours on a modern CPU. For a quick test, reduce to `N_REPETICIONES=5` and `N_ROUNDS=10`.







