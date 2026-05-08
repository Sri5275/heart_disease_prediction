# ❤️ Heart Disease Prediction — Learning Log

## 📌 Project Overview

This project focuses on predicting the presence of heart disease using the Cleveland dataset.
The workflow includes:

* Model building
* Evaluation
* Feature analysis
* Feature engineering
* Iterative improvement


# 📅 07-05-2026 — Initial Learnings

## 🤖 Logistic Regression Baseline

* Built a **Logistic Regression** model for heart disease prediction
* Achieved approximately **85% accuracy**
* Established a strong baseline model for comparison


## ⚠️ Feature Removal Experiment (`age`)

To understand feature importance, the `age` feature was intentionally removed.

### 🔬 Observation

* Model accuracy dropped significantly after removing `age`

### 📌 Insight

* `age` is a highly influential predictor in this dataset
* Removing important features can noticeably reduce performance


## 🧠 Key Takeaway

Feature importance can be validated through controlled feature-removal experiments.


## 🚀 Planned Improvements

### 🌲 Advanced Models

Planned experiments with:

* Random Forest Classifier
* Support Vector Machine (SVM)

### 🎯 Why?

These models can:

* Capture non-linear relationships
* Learn complex feature interactions better than Logistic Regression


### 🔧 Feature Engineering

Planned techniques:

* Interaction features
* Combining related variables
* Derived feature creation

### 🎯 Goal

Improve predictive power and model generalization.


# 📅 08-05-2026 — Today’s Learnings

## 🧪 Feature Dependency Experiment

### Experiment

* Removed the `age` feature
* Retrained the model

### 📉 Result

* Accuracy dropped significantly

### 📌 Conclusion

* Confirms that `age` is a critical feature for prediction


## 🛠️ Environment & Workflow Setup

### VS Code Improvements

* Installed Jupyter extension in VS Code
* Enabled notebook-style execution in `.py` files using:

```python
# %%
```

### 🎯 Benefits

* Inline plot visualization
* Easier debugging
* Markdown-supported documentation
* Faster experimentation workflow


# 📘 ML Concepts Learned

## 📊 Precision / Recall / F1-Score

### Precision

> “Out of predicted positives, how many were correct?”

### Recall

> “Out of actual positives, how many were found?”

### F1-Score

> Balance between Precision and Recall

### Support

> Number of samples in each class


## ⚖️ Macro Avg vs Weighted Avg

### Macro Average

* Treats all classes equally

### Weighted Average

* Weighted according to class size


## 🎯 `predict()` vs `predict_proba()`

### `predict()`

* Returns final predicted class (`0` or `1`)

### `predict_proba()`

* Returns probability/confidence scores

Example:

```python
[0.12, 0.88]
```

Meaning:

* 12% probability of Class 0
* 88% probability of Class 1


## 🎲 `random_state`

### What it does

Controls randomness in:

* Data shuffling
* Train-test splitting

### Key Insight

> `random_state = "Which shuffle of the dataset do I get?"`

### Important Notes

* Same value → same split every run
* Different values → different splits
* `42` is not mathematically special; it is just commonly used


## ⚖️ Why `stratify=y` Matters

### With Stratify

* Maintains original class distribution
* Produces balanced train/test sets
* Gives fair evaluation

### Without Stratify

* Class distribution may become skewed
* Evaluation can become biased


## 📊 Example: Stratified vs Non-Stratified Split

### ✅ With Stratify

* Accuracy: **0.81**
* ROC-AUC: **0.93**
* Support:

  * Class 0 → 32
  * Class 1 → 28

✔ Balanced evaluation


### ⚠️ Without Stratify

* Accuracy: **0.86** *(looks better)*
* ROC-AUC: **0.92**
* Support:

  * Class 0 → 36
  * Class 1 → 24

❌ Test distribution became skewed


## 🧠 Important Insight

Higher accuracy without stratification does **not** necessarily mean a better model.

Why?

* The test set became easier due to imbalance
* More negative samples increased chances of correct TN predictions

### Better interpretation:

> “The evaluation became biased, not the model smarter.”


## 📈 ROC-AUC

### What it measures

How well the model ranks positive vs negative samples using probabilities.

### Why it matters

* More stable than accuracy
* Less sensitive to class imbalance
* Evaluates ranking quality, not just hard predictions


## 🔧 `handle_unknown="ignore"`

### Purpose

Handles unseen categories during encoding.

### Benefits

* Prevents runtime errors
* Makes preprocessing pipelines robust
* Safely encodes unknown categories as zeros


# 📅 09-05-2026 — Planned Work

## 🤖 Model Improvement Strategy

### 🌲 Random Forest

* Explore non-linear modeling
* Capture feature interactions


### 🔧 Feature Engineering

Planned:

* New derived features
* Interaction terms
* Feature combinations


### 📊 Model Comparison

Compare:

* Logistic Regression
* Random Forest
* SVM


# 📊 Final Reflection

Machine Learning development is iterative:

> **Break → Analyze → Improve → Repeat**


# 🧠 Key Learnings

* Feature importance matters
* Accuracy alone is insufficient
* ROC-AUC provides deeper insight
* Stratified sampling improves fairness
* Probabilities are more informative than hard labels
* Robust preprocessing is essential
* Iterative experimentation improves understanding


# 💡 Overall Insight

This project strengthened understanding of:

* Feature dependency
* Evaluation metrics
* Probability-based prediction
* Data splitting fairness
* Model experimentation workflows
* Importance of preprocessing pipelines


# 🧾 One-Line Takeaway

> “Good ML is about fair data splitting, correct evaluation metrics, probability understanding, and robust preprocessing — not just accuracy.”
