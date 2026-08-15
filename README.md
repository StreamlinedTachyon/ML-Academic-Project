# Machine Learning Engineering Portfolio: Algorithms from Scratch & Advanced Optimization


# Machine Learning & Predictive Modeling Pipelines

This repository contains a collection of **end-to-end predictive modeling pipelines** and **from-scratch mathematical implementations** of core machine learning algorithms.

The project demonstrates:

1. **From-Scratch Algorithm Engineering:** Linear algebra and probabilistic mechanics implemented in pure NumPy without high-level wrappers ($K$-Means++, GMM via Expectation-Maximization, PCA with Spectral Whitening, LDA via Generalized Eigenvalue Decomposition, and Regularized Logistic Regression).
2. **Robust Data Engineering Pipelines:** Leak-free preprocessing pipelines covering outlier-resistant statistical imputation, custom Z-score standardization, and eigenvalue whitening across high-dimensional spatial and clinical datasets.
3. **Rigorous Optimization & Generalization:** 5-Fold Stratified Cross-Validation, exhaustive Grid Search tuning, Cost-Complexity Pruning ($ccp\_\alpha$), and multi-metric evaluation (Precision, Recall, F1-Score) to mitigate model variance.

---

## 📂 Repository Structure & Quick Links

| Module Focus | Jupyter Notebook Link |
| --- | --- |
| **Supervised Learning & Overfitting Mitigation** | [Decision-tree-SVM-code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Decision-tree-SVM-code.ipynb) |
| **GMM & Expectation-Maximization** | [GMM-for-clustering-and-outliar-detection.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/GMM-for-clustering-and-outliar-detection.ipynb) |
| **K-Means++ & Cluster Stability** | [K-means_code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/K-means_code.ipynb) |
| **PCA & LDA (From Scratch)** | [PCA-LDA Medical_Dataset.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/PCA-LDA%20Medical_Dataset.ipynb) |
| **High-Dim PCA & Naive Bayes** | [Naive-Bayes-Code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Naive-Bayes-Code.ipynb) |
| **Medical Logistic Regression (From Scratch)** | [Logistic-Regresison-Code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Logistic-Regresison-Code.ipynb) |
| **Distance Metric Optimization (KNN)** | [KNN_Geospatial.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/KNN_Geospatial.ipynb) |

---

## 📊 Comprehensive Benchmark Suite

---

### Module 1: Supervised Learning, Hyperparameter Tuning & Overfitting Mitigation

🔗 **[View Notebook: Decision-tree-SVM-code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Decision-tree-SVM-code.ipynb)**

* **Dataset:** California Housing Geospatial Benchmark (5,000 samples, 8 spatial/demographic features)
* **Task:** 3-Class Spatial Value Classification (`Low`, `Medium`, `High`)
* **Validation Scheme:** 5-Fold Stratified Cross-Validation (`StratifiedKFold`) + `GridSearchCV`

#### 1. Baseline Model Comparison & Execution Efficiency

| Model | Hyperparameter Grid | Optimization Time | Train F1 | Test F1 | Train/Test Gap | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | `C: [0.1, 1, 10]`, `penalty: ['l2']` | 4.80s | 0.7196 | 0.7048 | **0.0148** | Optimal Generalization |
| **Support Vector Machine (SVM)** | `C: [0.1, 1, 10]`, `kernel: ['linear', 'rbf']` | 2.83s | 0.8026 | 0.7215 | 0.0811 | Slight Overfitting |
| **Decision Tree (CART)** | `max_depth: [5, 10, 15]`, `min_samples_split: [2, 10, 20]` | **0.35s** | 0.8654 | 0.7115 | 0.1539 | High Overfitting |

#### 2. Advanced Cost-Complexity Pruning ($ccp\_\alpha$) on Decision Tree

To resolve the training data memorization of the baseline tree, minimal cost-complexity pruning was performed across the full pruning path:

* **Optimal Penalty Parameter:** `ccp_alpha = 0.0022`
* **Pre-Pruning:** Train F1: `0.8654` | Test F1: `0.7115` | Gap: `0.1539`
* **Post-Pruning:** Train F1: `0.7663` | Test F1: **`0.7276`** | Gap: **`0.0387`**
* **Result:** **74.8% reduction in generalization variance**, elevating test performance beyond the tuned RBF-SVM while maintaining sub-second training speed.

---

### Module 2: Unsupervised Clustering & Expectation-Maximization (From Scratch)

#### A. Gaussian Mixture Model (GMM) via Expectation-Maximization

🔗 **[View Notebook: GMM-for-clustering-and-outliar-detection.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/GMM-for-clustering-and-outliar-detection.ipynb)**

* **Formulation:** Probabilistic soft clustering optimizing mixture weights ($\pi_k$), cluster centroids ($\mu_k$), and full covariance matrices ($\Sigma_k$) with Tikhonov damping ($\epsilon = 10^{-6}\mathbf{I}$) to guarantee positive semi-definiteness.
* **Convergence Threshold:** Absolute log-likelihood change $\Delta \mathcal{L} < 10^{-4}$.

| Iteration | Log-Likelihood ($\mathcal{L}$) | Step $\Delta \mathcal{L}$ | Cumulative $\Delta \mathcal{L}$ Gain | Convergence Status |
| --- | --- | --- | --- | --- |
| **Iter 1** | `-343.2978` | — | — | Initial Assignment |
| **Iter 10** | `-205.2954` | $+0.5358$ | $+138.0024$ | Active EM Ascent |
| **Iter 20** | `-180.5395` | $+0.6364$ | $+162.7583$ | Near Plateau |
| **Iter 28** | **`-180.1855`** | $< 10^{-4}$ | **$+163.1123$** | **Optimal Convergence** |

* **Anomaly Detection:** Evaluated sample log-probabilities via `score_samples()`; an enforced 5th-percentile cutoff threshold identified **8 distinct boundary anomalies**.

#### B. $K$-Means++ Clustering with Geometric Optimization

🔗 **[View Notebook: K-means_code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/K-means_code.ipynb)**

* **Algorithm:** Vectorized Expectation-Maximization with probabilistic distance-weighted initialization ($D(x)^2 / \sum D(x)^2$).
* **Custom vs. Production Benchmark ($k = 6$):** The Custom NumPy Engine achieved a WCSS of 17,162.85 in just 0.0687s, coming within $\Delta = 0.17$ of `scikit-learn`'s C/Cython implementation, proving strict mathematical accuracy.

| Cluster Count ($k$) | WCSS (Inertia) | Mean Silhouette Score | Cluster Geometry Verdict |
| --- | --- | --- | --- |
| $k = 2$ | 24,891.12 | 0.2841 | Under-partitioned |
| **$k = 3$** | **20,114.65** | **0.3421** | **Optimal Global Cohesion & Separation** |
| $k = 4$ | 18,920.40 | 0.2910 | Marginal boundary overlap |

---

### Module 3: Dimensionality Reduction, Scatter Optimization & Whitening (From Scratch)

#### A. Principal Component Analysis (PCA) & Spectral Whitening

🔗 **[View Notebook: PCA-LDA Medical_Dataset.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/PCA-LDA%20Medical_Dataset.ipynb)**

* **Mathematical Pipeline:** Covariance Matrix ($\Sigma$) $\rightarrow$ Eigen-Decomposition ($\mathbf{\Lambda}, \mathbf{V}$) $\rightarrow$ Spectral Whitening Transformation ($X_{\text{whitened}} = \frac{X_{\text{centered}} \cdot \mathbf{V}_k}{\sqrt{\mathbf{\Lambda}_k + \epsilon}}$)

| Dataset | Raw Dimensions | Imputation Strategy | Target Subspace | Total Retained Variance |
| --- | --- | --- | --- | --- |
| **Breast Cancer** | 30 | 5% Synthetic MCAR (Mean Imputed) | **2 Components** | **60.48%** (PC1: 42.11%, PC2: 18.37%) |

#### B. Linear Discriminant Analysis (LDA) from Scratch

* **Discriminative Efficiency:** Evaluated the $30 \times 30$ Within-Class Scatter Matrix ($S_w$) and Between-Class Scatter Matrix ($S_b$). Compressed the space into a **1D Linear Discriminant (`LD1`)**, capturing **100.00% of the class-separability variance**.

#### C. High-Dimensional Arrhythmia Diagnostic Pipeline

🔗 **[View Notebook: Naive-Bayes-Code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Naive-Bayes-Code.ipynb)**

* **Preprocessing:** Reduced 279 clinical & ECG features (408 missing entries median-imputed, 17 zero-variance channels dropped).
* **PCA Compression:** Subspace retained **66.50% variance** across 30 principal components.
* **Supervised Classification:** 5-Fold Stratified `GridSearchCV` on Gaussian Naive Bayes (`var_smoothing = 0.1`) yielded an isolated Holdout F1-Score of **0.6582** (Precision: 0.7027).

---

### Module 4: Medical Supervised Learning Pipeline (From Scratch)

🔗 **[View Notebook: Logistic-Regresison-Code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Logistic-Regresison-Code.ipynb)**

* **Dataset:** Parkinson's Disease Acoustic Dataset (UCI).
* **Profile:** 195 Patient Samples, 22 Biomechanical Acoustic Features (75.38% Positive / 24.62% Negative).
* **Architecture:** Custom Logistic Regression with Binary Cross-Entropy Loss and Analytical $L_2$ Regularization. A custom, leak-free **Z-Score Standardization** engine and **Stratified 5-Fold Split Generator** were programmed from the ground up.

#### Grid Search & Diagnostic Holdout Performance

* **Grid Sweep:** $\alpha \in [0.01, 0.1, 0.5]$, $\lambda \in [0.01, 0.1, 1.0]$. Optimal configuration found at `Learning Rate = 0.5`, `Lambda = 0.01` with a peak CV F1-Score of **0.9132**.

| Diagnostic Metric | Score | Clinical Interpretation |
| --- | --- | --- |
| **Loss Convergence** | **$0.6932 \rightarrow 0.2689$** | **61.2% cost reduction** via stable analytical gradient descent. |
| **Precision** | **0.8961** | Low false positive rate; high certainty in clinical diagnosis. |
| **Recall (Sensitivity)** | **0.9388** | Minimizes false negatives in patient screening. |
| **F1-Score** | **0.9169** | Robust harmonic balance on imbalanced clinical data. |

---

### Supplementary Module: Distance Metric Optimization

🔗 **[View Notebook: KNN_Geospatial.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/KNN_Geospatial.ipynb)**

* **Task:** Evaluated the impact of spatial distance metrics on lazy-learning classification algorithms (K-Nearest Neighbors).
* **Results:** A robust `GridSearchCV` workflow established that Euclidean ($p=2$) peaked at $k=12$, while Manhattan ($L_1$) shifted the optimal neighborhood boundary to $k=13$, both achieving **100% test accuracy** while demonstrating how distance topology influences optimal neighborhood density.

---

## 🛠️ Skills & Technology Matrix

| Core Competency | Specific Implementations |
| --- | --- |
| **Mathematical ML (From Scratch)** | $K$-Means++, GMM (EM Algorithm), PCA (Eigenvalue Decomposition), LDA (Scatter Matrices), Logistic Regression (Analytical Gradients), Custom Z-Score Scaler. |
| **Data Pipelines & Preprocessing** | Median/Mean Statistical Imputation, Zero-Variance Channel Filtering, Spectral Whitening, Multi-Class Quantile Binning. |
| **Model Optimization** | Stratified $K$-Fold Cross-Validation, GridSearchCV, Cost-Complexity Pruning ($ccp\_\alpha$), SAGA Solver Regularization ($L_1/L_2$). |
| **Diagnostic Evaluation** | Macro/Micro F1-Score, Precision-Recall Tradeoffs, Confusion Matrix Decomposition, Silhouette Analysis, WCSS/Elbow Tracking. |
| **Libraries & Frameworks** | Python 3.10+, NumPy, Pandas, scikit-learn, SciPy, Matplotlib, Seaborn. |

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

