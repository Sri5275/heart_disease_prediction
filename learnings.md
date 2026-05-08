# 📊 Classification Metrics (ML Evaluation)

These metrics are commonly used to evaluate classification models in scikit-learn.

## 🎯 Precision

**Definition:**
Out of all predicted positive cases, how many were actually correct?

**Formula:**

```
Precision = TP / (TP + FP)
```

**Interpretation:**

* Focuses on **correctness of positive predictions**
* High precision → fewer false positives

## 🔍 Recall (Sensitivity / True Positive Rate)

**Definition:**
Out of all actual positive cases, how many did the model correctly identify?

**Formula:**

```
Recall = TP / (TP + FN)
```

**Interpretation:**

* Focuses on **capturing all positive cases**
* High recall → fewer false negatives

## ⚖️ F1-Score

**Definition:**
Harmonic mean of precision and recall.

**Formula:**

```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretation:**

* Balances precision and recall
* Useful when there is class imbalance

## 📦 Support

**Definition:**
Number of actual samples in each class.

**Interpretation:**

* Shows dataset distribution
* Helps understand class imbalance

## 📊 Macro Average

**Definition:**
Simple average of metrics across all classes.

**How it works:**

* Each class is treated equally
* No weighting based on sample size

**Use case:**

* Useful when all classes are equally important

## ⚖️ Weighted Average

**Definition:**
Average of metrics weighted by the number of samples in each class.

**How it works:**

* Larger classes have more influence

**Use case:**

* Better for imbalanced datasets

## 🧠 Summary

* **Precision → correctness of positive predictions**
* **Recall → completeness of positive detection**
* **F1-score → balance between precision & recall**
* **Support → dataset distribution**
* **Macro avg → equal importance to all classes**
* **Weighted avg → importance based on class size**

