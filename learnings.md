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


---


# Why Stratified Sampling Matters in Model Evaluation??

## ⚠️ Why “without stratify” is tricky

Without `stratify`, the **class distribution in train/test split can change randomly**.

* The test set may contain more of one class and fewer of another
* This creates a **biased evaluation**, because the test data is no longer representative of the full dataset


## 📊 Example Results

*(You can verify this using the `classification_report` — especially the support values for each class)*

### ✅ With stratify

* Accuracy: **0.81**
* ROC-AUC: **0.93**
* ✔ Balanced and fair evaluation
* Support: **Class 0 = 32**, **Class 1 = 28**


### ⚠️ Without stratify

* Accuracy: **0.86** (looks better ❗)
* ROC-AUC: **0.92**
* ❌ Skewed test distribution
* Support: **Class 0 = 36**, **Class 1 = 24**


## 🧠 What “more negatives” means

Here, **“more negatives” means:**

👉 The test set accidentally has a **higher proportion of Class 0 samples** than the original dataset.

So instead of a balanced test set:

* Class 0 (negative class) becomes **overrepresented**
* Class 1 (positive class) becomes **underrepresented**


## 📊 Why this matters

When the test set has more negatives:

* ✔ Model gets more chances to predict **TN (True Negatives)** correctly
* ✔ FN (False Negatives) may decrease simply because there are fewer positives
* ⚠ Accuracy may look better, but it is misleading

But:

👉 This improvement is **not because the model improved**
👉 It is because the **evaluation became easier due to imbalance**


## 🎯 Final Insight

👉 **Stratified split is better**

Because it ensures:

* Balanced representation of classes
* Fair and realistic evaluation
* Reliable comparison of model performance


## 🧾 One-line intuition

* **Accuracy** = “Did I guess right?”
* **ROC-AUC** = “Did I rank probabilities correctly?”
* **Stratify** = “Did I test the model on a fair distribution?”

# 📘 Feature Importance, Permutation Importance & PDP (Crisp Notes)


# 1. Feature Importance

Tells:
> Which features matter most to the model.

### Types:
- **Tree-based models**: importance from splits
- **Linear models**: based on coefficients

### Limitation:
- Can be biased
- Affected by correlated features


# 2. Permutation Importance

Library: :contentReference[oaicite:0]{index=0}

Tells:
> How much model performance drops when a feature is shuffled.

### Steps:
1. Train model
2. Get baseline score
3. Shuffle one feature
4. Measure score drop

### Interpretation:
- Big drop → important feature
- Small drop → unimportant feature

### Pros:
- Model-agnostic
- Reliable

### Cons:
- Slow
- Sensitive to correlated features


# 3. Partial Dependence Plot (PDP)

Tells:
> How a feature affects model predictions on average.

### How it works:
- Fix one feature value
- Average predictions across dataset

### Interpretation:
- Upward curve → increases prediction
- Flat → no effect
- Non-linear → complex relationship

### Pros:
- Easy visualization
- Shows trend

### Cons:
- Assumes feature independence
- Misses feature interactions


# ⚖️ Quick Comparison

| Method | Type | Answer |
|--------|------|--------|
| Feature Importance | Global | What matters? |
| Permutation Importance | Global | What truly affects performance? |
| PDP | Global | How does feature affect prediction? |


# 🧠 Final Insight

- Feature Importance → importance estimate
- Permutation Importance → real impact test
- PDP → behavior visualization