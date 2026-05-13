# ❤️ Heart Disease ML Project — Learning Index

## 📌 Overview

This document tracks key learnings from building a heart disease prediction system using Machine Learning.


# 📚 Learning Log


## 🧠 Day 1 — Baseline Model & Evaluation

| Topic | Key Learning |
|------|--------------|
| Logistic Regression | Built first baseline model (~85% accuracy) |
| Feature Dependency Test | Removing `age` reduced accuracy → strong feature impact |
| Evaluation Metrics | Learned precision, recall, F1-score, confusion matrix |
| Train-Test Split | `stratify=y` ensures balanced class distribution |
| Prediction Types | `predict()` vs `predict_proba()` difference |
| ROC-AUC | Better metric than accuracy for imbalanced data |


## 🧠 Day 2 — Workflow & Data Handling

| Topic | Key Learning |
|------|--------------|
| VS Code Workflow | Used `# %%` for notebook-style execution |
| Random State | Ensures reproducible splits |
| Data Leakage | Must scale AFTER train-test split |
| Stratification Impact | Prevents biased evaluation |
| Encoding | `handle_unknown="ignore"` improves robustness |


## 🧠 Day 3 — Model Explainability & Interpretation

| Topic | Key Learning |
|------|--------------|
| Feature Importance | Identifies most influential features |
| Permutation Importance | Measures real impact via performance drop |
| Partial Dependence Plot (PDP) | Shows feature effect on predictions |
| eli5 Library | Used for model interpretability |
| show_prediction vs show_weights | Local vs global explanation difference |
| Model Debugging | Focus on understanding behavior, not just accuracy |


## 🧠 Day 4 — System Design Thinking

| Topic | Key Learning |
|------|--------------|
| ML as System | ML is a pipeline, not just a model |
| Modularity | Separate ingestion, training, evaluation |
| Reproducibility | Systems must produce same results consistently |
| Experimentation Loop | Break → Analyze → Improve → Repeat |
| Engineering Mindset | Focus on reliability + scalability |


# 🧠 Key Concepts Summary

- Feature importance ≠ causation
- Permutation importance = real performance impact
- PDP = feature behavior visualization
- Accuracy alone is misleading
- ROC-AUC is more reliable for imbalance
- Stratified splitting ensures fairness
- Probabilities matter more than hard labels
- ML systems must be modular and reproducible

# 🔥 Final Insight

> “Machine Learning is not model building — it is system building with continuous validation, interpretation, and improvement.”