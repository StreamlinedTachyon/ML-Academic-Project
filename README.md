# Machine Learning Engineering Portfolio: Algorithms from Scratch & Advanced Optimization

Welcome to my Machine Learning Portfolio. This repository contains a collection of Jupyter Notebooks demonstrating both **first-principles mathematical algorithm development** (built entirely from scratch using pure NumPy) and **production-grade ML pipelines** (using `scikit-learn`).

I designed this repository to demonstrate my deep understanding of the mathematical engines powering modern AI, as well as my practical ability to optimize models, evaluate complex metrics, and aggressively mitigate overfitting.

---

## 🛠 Key Skills Demonstrated

* **Algorithms From Scratch:** Gaussian Mixture Models (EM Algorithm), Principal Component Analysis (Eigen Decomposition & Whitening), Linear Discriminant Analysis (Scatter Matrices), and K-Means++ (Probabilistic Initialization).
* **Supervised Learning Pipelines:** Support Vector Machines, Logistic Regression, Decision Trees, and K-Nearest Neighbors.
* **Overfitting Mitigation:** Cost-Complexity Pruning (CCP), L2 Regularization, and K-Fold Cross-Validation.
* **Data Engineering:** Manual Z-score Standardization, Data Imputation, and Spatial Feature Engineering.
* **Evaluation & Analytics:** Silhouette Scores, WCSS, Log-Likelihood convergence, Confusion Matrices, and Macro F1-Score optimization.

---

## 📂 Repository Structure & Projects Overview

### 1. Geospatial Classification & Cost-Complexity Pruning

**File:** `01_Geospatial_Classification_and_Pruning.ipynb`

* **Objective:** Classify California Housing data into Low, Medium, and High-value zones using spatial and demographic features.
* **Highlights:**
* Implemented an end-to-end `scikit-learn` Pipeline with `StratifiedKFold` and `GridSearchCV`.
* Systematically benchmarked Logistic Regression, Decision Trees, and SVMs using Macro F1-Scores.
* Conducted an **Aggressive Overfitting Analysis** to measure the Train/Test performance gap.
* **Advanced Mitigation:** Extracted the pruning path of an overfitted Decision Tree and implemented **Cost-Complexity Pruning (ccp_alpha)** to collapse the Train/Test gap and maximize generalization.



### 2. Gaussian Mixture Model (GMM) & Anomaly Detection From Scratch

**File:** `02_GMM_From_Scratch.ipynb`

* **Objective:** Build a probabilistic clustering engine without using high-level ML libraries.
* **Highlights:**
* Implemented the **Expectation-Maximization (EM)** algorithm entirely in `numpy`.
* Manually programmed the E-Step (Responsibility calculation via Multivariate Gaussian PDF) and M-Step (Mean, Covariance, and Weight updates).
* Monitored mathematical convergence via Log-Likelihood tracking.
* Developed a probabilistic **Outlier Detection** system by isolating data points falling below the 5th percentile of the probability density threshold.



### 3. Dimensionality Reduction (PCA & LDA) & K-Means++ From Scratch

**File:** `03_DimReduction_and_KMeans_Scratch.ipynb`

* **Objective:** Reduce the dimensionality of a high-dimensional medical dataset (Breast Cancer) and build an optimized K-Means engine.
* **Highlights:**
* **PCA From Scratch:** Calculated Covariance matrices, sorted Eigenvalues/Eigenvectors, and implemented Eigenvalue Decomposition (Whitening).
* **LDA From Scratch:** Computed Within-Class and Between-Class Scatter Matrices to project data maximizing class separability.
* **K-Means++:** Implemented proportional probability distance algorithms `D(x)^2` to smartly initialize centroids and avoid local minima.
* **Engineering Benchmark:** Benchmarked the custom NumPy K-Means against `scikit-learn`'s C-level Cython implementation, proving exact mathematical parity in WCSS (Within-Cluster Sum of Squares) scores.



### 4. K-Nearest Neighbors (KNN) Custom CV & Hyperparameter Tuning

**File:** `04_KNN_Optimization.ipynb`

* **Objective:** Build a robust validation pipeline for instance-based learning.
* **Highlights:**
* Programmed a manual K-Fold Cross-Validation loop to evaluate `k` values from 1 to 30.
* Visualized the Cross-Validated Error Rate to identify the optimal number of neighbors.
* Upgraded the distance metric to `Manhattan` distance and validated improvements using `GridSearchCV`.



---

## ⚙️ Prerequisites & Installation

To run the notebooks locally, you will need Python 3.8+ and the following libraries installed.

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-portfolio.git
cd ml-portfolio

# Install required dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy

```

---

## 🚀 How to Run

Simply start a Jupyter Notebook server in the root directory:

```bash
jupyter notebook

```

Navigate to any of the `.ipynb` files. All cells are pre-formatted to print execution times, convergence metrics, and evaluation reports directly to the standard output.

---

## 💡 Note to Interviewers

While using high-level APIs like `.fit()` and `.predict()` is essential for rapid development, I built the **"From Scratch"** notebooks in this repository to prove my grasp of the underlying Linear Algebra and Calculus that power these frameworks.

By avoiding black-box implementations where appropriate, I am able to debug complex convergence issues, prevent data leakage in pipelines, and write highly optimized, vectorized code.

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

