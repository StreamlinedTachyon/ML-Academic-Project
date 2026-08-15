# 🚀 Machine Learning Engineering Portfolio: Algorithms from Scratch & Advanced Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange.svg)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%26%20CV-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Benchmarks-Verified-brightgreen.svg)]()

## 📌 Note to the Evaluation Panel
Welcome to my Machine Learning Engineering portfolio. In mission-critical environments, treating AI as a "black box" is a liability. This repository was designed to demonstrate my ability to develop **core machine learning algorithms from first mathematical principles** (pure NumPy) alongside **production-grade optimized pipelines**. 

By engineering these systems from the ground up, I ensure strict data isolation, complete control over mathematical convergence, and aggressive mitigation of model overfitting—resulting in highly reliable, generalized predictive systems.

---

## 🛠️ Core Competencies & Technology Matrix

| Engineering Focus | Specific Implementations |
| :--- | :--- |
| **Mathematical ML (From Scratch)** | $K$-Means++, GMM (EM Algorithm), PCA (Eigenvalue Decomposition), LDA (Scatter Matrices), Logistic Regression (Analytical Gradients), Custom Z-Score Scaler. |
| **Data Pipelines & Preprocessing** | Median/Mean Statistical Imputation, Zero-Variance Channel Filtering, Spectral Whitening, Multi-Class Quantile Binning. |
| **Model Optimization** | Stratified $K$-Fold Cross-Validation, GridSearchCV, Cost-Complexity Pruning ($ccp\_\alpha$), SAGA Solver Regularization ($L_1/L_2$). |
| **Diagnostic Evaluation** | Macro/Micro F1-Score, Precision-Recall Tradeoffs, Confusion Matrix Decomposition, Silhouette Analysis, WCSS/Elbow Tracking. |

---

## 📂 Repository Index & Technical Benchmarks

The repository is divided into distinct operational modules. Each module targets a specific data challenge, from high-dimensional signal extraction to geospatial classification.

### Module 1: Supervised Learning, Hyperparameter Tuning & Overfitting Mitigation
🔗 **[View Source Code: Decision-tree-SVM-code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Decision-tree-SVM-code.ipynb)**

*   **Objective:** Classify California Housing data into geospatial value zones while aggressively targeting and reducing training variance.
*   **Validation Scheme:** 5-Fold Stratified Cross-Validation + `GridSearchCV`.

**Baseline Execution & Efficiency Benchmark:**

| Model | Optimization Time | Train F1 | Test F1 | Train/Test Gap | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 4.80s | 0.7196 | 0.7048 | **0.0148** | Optimal Generalization |
| **Support Vector Machine** | 2.83s | 0.8026 | 0.7215 | 0.0811 | Slight Overfitting |
| **Decision Tree (CART)** | **0.35s** | 0.8654 | 0.7115 | 0.1539 | High Overfitting |

**Advanced Cost-Complexity Pruning ($ccp\_\alpha$):**
To resolve the baseline Decision Tree's data memorization, minimal cost-complexity pruning was performed. 
*   **Post-Pruning Metric:** Applying an optimal penalty of `ccp_alpha = 0.0022` collapsed the Train/Test gap from `0.1539` down to **`0.0387`**. 
*   **Result:** Generalization variance was reduced by **74.8%**, allowing the Tree to outperform the SVM on unseen test data while maintaining sub-second training speeds.

---

### Module 2: Unsupervised Clustering & Probabilistic Mechanics (From Scratch)

**A. Gaussian Mixture Model (GMM) via Expectation-Maximization**
🔗 **[View Source Code: GMM-for-clustering-and-outliar-detection.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/GMM-for-clustering-and-outliar-detection.ipynb)**
*   **Mechanics:** Pure NumPy implementation optimizing mixture weights ($\pi_k$), centroids ($\mu_k$), and full covariance matrices ($\Sigma_k$) with Tikhonov damping ($\epsilon = 10^{-6}\mathbf{I}$) to prevent singularity.
*   **Convergence:** Achieved strict monotonic log-likelihood maximization, converging at $\mathcal{L} =$ **`-180.1855`** in exactly 28 iterations.
*   **Anomaly Detection:** Evaluated sample log-probabilities via `score_samples()`. An enforced 5th-percentile cutoff threshold successfully flagged **8 distinct boundary anomalies**.

**B. $K$-Means++ Clustering with Geometric Optimization**
🔗 **[View Source Code: K-means_code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/K-means_code.ipynb)**
*   **Mechanics:** Vectorized EM with probabilistic distance-weighted initialization ($D(x)^2 / \sum D(x)^2$).
*   **Engine Benchmark:** The Custom NumPy Engine computed a Within-Cluster Sum of Squares (WCSS) of 17,162.85 in just 0.0687s, coming within $\Delta = 0.17$ of `scikit-learn`'s production C-level implementation, verifying strict mathematical accuracy.

---

### Module 3: Dimensionality Reduction & Signal Extraction (From Scratch)

**A. Principal Component Analysis (PCA) & Linear Discriminant Analysis (LDA)**
🔗 **[View Source Code: PCA-LDA Medical_Dataset.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/PCA-LDA%20Medical_Dataset.ipynb)**
*   **Methodology:** Hand-coded Covariance Matrix construction $\Sigma = \frac{1}{N-1}X^T X$, Eigen-Decomposition, and Spectral Whitening Transformations. For LDA, solved the generalized eigenvalue problem $S_w^{-1} S_b \mathbf{w} = \lambda \mathbf{w}$.
*   **Result (Breast Cancer Dataset):** PCA captured **60.48% variance** in just 2 components. LDA compressed the 30-D space into a 1D discriminant capturing **100.00% of class-separability variance**.

**B. High-Dimensional Arrhythmia Diagnostic Pipeline**
🔗 **[View Source Code: Naive-Bayes-Code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Naive-Bayes-Code.ipynb)**
*   **Preprocessing:** Outlier-resistant median statistical imputation (resolved 408 missing entries) and zero-variance channel filtering on 279-D ECG features. 
*   **Result:** Reduced dimensions by **88.5%** (retaining 66.50% variance). The downstream Gaussian Naive Bayes model (`var_smoothing = 0.1`) maintained a robust **0.7027 Precision** on an unseen holdout set.

---

### Module 4: Medical Supervised Learning & Logistic Regression (From Scratch)
🔗 **[View Source Code: Logistic-Regresison-Code.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/Logistic-Regresison-Code.ipynb)**

*   **Task:** Parkinson's Disease acoustic diagnostic classification (75.38% Positive / 24.62% Negative class imbalance).
*   **Architecture:** Custom Logistic Regression utilizing Binary Cross-Entropy Loss, Analytical $L_2$ Regularization, and a leak-free **Z-Score Standardization** engine programmed from scratch.

**Diagnostic Holdout Performance (Optimal: $\alpha = 0.5$, $\lambda = 0.01$):**

| Diagnostic Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Loss Convergence** | **$0.6932 \rightarrow 0.2689$** | **61.2% cost reduction** via stable analytical gradient descent. |
| **Precision** | **0.8961** | Low false positive rate; high certainty in clinical diagnosis. |
| **Recall (Sensitivity)** | **0.9388** | Minimizes false negatives in patient screening. |
| **F1-Score** | **0.9169** | Robust harmonic balance mitigating class imbalance. |

---

### Module 5: Distance Metric Topology Optimization (KNN)
🔗 **[View Source Code: KNN_Geospatial.ipynb](https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking/blob/main/KNN_Geospatial.ipynb)**

*   **Objective:** Evaluate the impact of spatial distance metrics on lazy-learning classification.
*   **Result:** `GridSearchCV` established that Euclidean distance ($p=2$) peaked at $k=12$, while Manhattan distance ($L_1$) shifted the optimal neighborhood boundary to $k=13$. Both distance topologies successfully sustained **100% test accuracy**.

---
## 🚀 Environment Setup & Execution

To run these pipelines and verify the mathematical benchmarks locally:

```bash
# 1. Clone and navigate into the repository
git clone https://github.com/StreamlinedTachyon/Machine-Learning-Algorithm-Optimization-and-Benchmarking.git
cd Machine-Learning-Algorithm-Optimization-and-Benchmarking

# 2. Create and activate virtual environment
python -m venv venv

# Linux / macOS:
source venv/bin/activate

# Windows (CMD / PowerShell):
# venv\Scripts\activate

# 3. Install dependencies and start Jupyter
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter notebook
```
