# 📊 Machine Learning Metrics Cheat Sheet

A quick reference of evaluation metrics across different types of machine learning tasks in scikit-learn.


# 🎯 1. Classification Metrics

Used for predicting categories (e.g., yes/no, disease/no disease).

## 📌 Core Metrics

* **Accuracy** → Overall correctness

  ```txt
  (TP + TN) / Total
  ```

* **Precision** → Correctness of positive predictions

  ```txt
  TP / (TP + FP)
  ```

* **Recall (Sensitivity)** → Ability to find actual positives

  ```txt
  TP / (TP + FN)
  ```

* **F1-Score** → Balance between precision & recall

  ```txt
  2 * (P * R) / (P + R)
  ```

* **Support** → Number of samples per class


## 📊 Advanced Metrics

* **ROC-AUC** → Class separation quality
* **Log Loss** → Prediction confidence error
* **Cohen’s Kappa** → Agreement beyond chance
* **MCC (Matthews Correlation Coefficient)** → Balanced metric for imbalanced data


## 🧩 Confusion Matrix

```txt
                 Predicted
               0         1
Actual 0     TN        FP
Actual 1     FN        TP
```


# 📉 2. Regression Metrics

Used for predicting continuous values (e.g., price, temperature).

## 📌 Error Metrics

* **MAE (Mean Absolute Error)**

  ```txt
  avg(|y - ŷ|)
  ```

* **MSE (Mean Squared Error)**

  ```txt
  avg((y - ŷ)²)
  ```

* **RMSE (Root MSE)**

  ```txt
  √MSE
  ```


## 📊 Fit Metrics

* **R² Score** → Variance explained by model

  ```txt
  1 - (SS_res / SS_tot)
  ```

* **Adjusted R²** → Penalized R² for feature count


# 🧠 3. Clustering Metrics

Used for unsupervised learning (no labels).

## 📌 Internal Metrics

* **Silhouette Score** → Cluster separation quality
* **Davies–Bouldin Index** → Lower is better clustering
* **Calinski–Harabasz Index** → Higher is better clustering


## 📌 External Metrics (if labels exist)

* **ARI (Adjusted Rand Index)**
* **NMI (Normalized Mutual Information)**
* **Fowlkes–Mallows Score**


# 🤖 4. Deep Learning Metrics

Used in neural networks and large models.

## 📌 Classification

* Accuracy
* Precision / Recall / F1
* AUC-ROC


## 📌 Training Metrics

* Training Loss
* Validation Loss
* Epoch Accuracy
* Overfitting Gap


## 📌 Loss Functions

* Cross-Entropy Loss (classification)
* MSE Loss (regression)
* Huber Loss (robust regression)
* KL Divergence (probability matching)


## 📌 Task-specific Metrics

* **BLEU** → Machine translation
* **ROUGE** → Text summarization
* **Perplexity** → Language models
* **IoU / Dice Score** → Image segmentation


# 🔍 5. Ranking / Recommendation Metrics

* **Precision@K** → Relevant items in top K
* **Recall@K** → Coverage in top K
* **MAP (Mean Average Precision)** → Ranking quality
* **NDCG** → Position-aware ranking quality
* **MRR** → First relevant result quality


# 🧠 Quick Decision Guide

```txt
Classification → F1 / Accuracy / ROC-AUC
Regression     → RMSE / MAE / R²
Clustering     → Silhouette Score
Deep Learning  → Loss + task metric
Ranking        → NDCG / MAP
```


# 🚀 Summary

* Classification → correctness of categories
* Regression → prediction error
* Clustering → structure of groups
* Deep Learning → loss + task metrics
* Ranking → position-based relevance
