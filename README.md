# ML-Academic-Project

A curated collection of hands-on Jupyter notebooks that demonstrate core machine learning algorithms, data-processing techniques, and evaluation/visualization examples for academic learning and quick experimentation. These notebooks use Python and scikit-learn-based workflows to show classification, regression, clustering, distance metrics, and data-handling approaches on small example datasets (including the Iris dataset and included dummy CSVs).

* **Language:** Python (notebooks)
* **Notebooks tested with:** Python 3.12 (see individual notebook metadata)
* **License:** MIT (LICENSE file included)

## Table of contents
- [About](#about)
- [What’s in this repository (high-level)](#whats-in-this-repository-high-level)
- [Notebook index (short descriptions)](#notebook-index-short-descriptions)
- [Requirements](#requirements)
- [Quick start — run a notebook](#quick-start--run-a-notebook)
- [How it’s organized](#how-its-organized)
- [How it fits together](#how-it-fits-together)
- [Recommended workflow / best practices](#recommended-workflow--best-practices)
- [Suggested next steps](#suggested-next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About
This repo is intended as an educational resource and demo playground for ML concepts:

* **Classification algorithms** (KNN, Naive Bayes, Logistic Regression, Decision Trees, SVM)
* **Regression and optimization** (Gradient Descent)
* **Unsupervised learning** (K-Means, GMM clustering and outlier detection)
* **Dimensionality reduction** (PCA, LDA)
* **Distance measures and dynamic time warping** examples
* **Data preprocessing** and missing-value handling
* **Visualization** and model-evaluation examples

Each notebook is standalone and demonstrates a single concept or experiment with runnable code and plotted outputs.

## What’s in this repository (high-level)

**Top-level notebooks and folders (representative):**
* `Decision-tree-SVM-code.ipynb`
* `GMM-for-clustering-and-outlier-detection.ipynb`
* `K-means_code.ipynb`
* `KNN-code.ipynb`
* `Logistic-Regression-Code.ipynb`
* `Naive-Bayes-Code.ipynb`
* `PCA-and-LDA.ipynb`
* PCA / LDA, model-evaluation and visualization notebooks

**Directories:**
* **Classification/**
  * `multiclass-classification.ipynb` — Iris + GaussianNB example and metrics
  * `model_evaluation_plots.ipynb` — underfitting vs overfitting visualization
* **Regression/**
  * `Regression.ipynb` — regression examples and experiments
  * `Gradient-Descent.ipynb` — gradient descent walkthrough/visualization
* **Data processing-handling missing values/**
  * `Data-handling-code.ipynb` — missing-value handling and simple preprocessing
  * `dummy_dataset (1).csv` — example dataset for preprocessing notebook
* **Distance_metrics_and_dynamic_time_warping/**
  * `Linking_Euclidean_Mahalanobis.ipynb`
  * `distance-measures.ipynb`
* **Iris_Gaussian_Distribution/**
  * `Iris.csv` — dataset used by the Iris notebooks
  * `Multivariate-Gaussian-Distribution (1).ipynb`
* **License file:** `LICENSE` (MIT)

## Notebook index (short descriptions)
*(Use these to pick which notebook to open first.)*

* **`KNN-code.ipynb`** — k-Nearest Neighbors classifier, basic usage and evaluation.
* **`Naive-Bayes-Code.ipynb`** — Gaussian Naive Bayes example on small datasets.
* **`Logistic-Regression-Code.ipynb`** — logistic regression classification demonstration.
* **`Decision-tree-SVM-code.ipynb`** — decision trees and SVM classifier comparisons.
* **`K-means_code.ipynb`** — K-Means clustering demonstration and visualizations.
* **`GMM-for-clustering-and-outlier-detection.ipynb`** — Gaussian Mixture Models for clustering and outlier detection.
* **`PCA-and-LDA.ipynb`** — dimensionality reduction: PCA and LDA examples.
* **`Classification/multiclass-classification.ipynb`** — Iris dataset split, training, metrics (confusion matrix, precision/recall/F1).
* **`Classification/model_evaluation_plots.ipynb`** — underfitting vs overfitting illustration and plotting.
* **`Regression/Regression.ipynb`** — regression modeling examples and analysis.
* **`Regression/Gradient-Descent.ipynb`** — gradient descent mechanics and plots.
* **`Data processing-handling missing values/Data-handling-code.ipynb`** — handling missing values, simple imputation examples.
* **`Distance_metrics_and_dynamic_time_warping/distance-measures.ipynb`** — distance metric comparisons (Euclidean, Mahalanobis, etc.)
* **`Iris_Gaussian_Distribution/Multivariate-Gaussian-Distribution (1).ipynb`** — Gaussian distribution / likelihood examples on Iris

## Requirements
These notebooks rely on the standard Python scientific stack. A minimal set of packages:

* Python 3.8+ (notebooks show Python 3.12 metadata)
* jupyter (or jupyterlab / notebook)
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn
* scipy

If you want reproducible env management, create a `requirements.txt` or `environment.yml`. Example pip list (run from an activated venv):
```bash
pip install jupyter numpy pandas scikit-learn matplotlib seaborn scipy
