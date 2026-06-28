# 🏦 Bank Customer Churn Prediction

**Preeti Bhardwaj** | [github.com/mistyvisty](https://github.com/mistyvisty)

End-to-end ML pipeline predicting which bank customers are likely to churn, with business-oriented threshold tuning and SHAP explainability for stakeholder communication.

---

## 🎯 Problem

Customer churn costs banks significantly more than retention. This model identifies at-risk customers before they leave, allowing targeted intervention. Built on a real banking dataset with class imbalance handled rigorously inside the pipeline.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test ROC-AUC | **0.847** |
| Mean CV AUC (5-fold) | **~0.87** |
| Model | XGBoost |
| Validation | Stratified 5-Fold Cross-Validation |

---

## ⚙️ Pipeline

Raw data → EDA → Feature engineering → SMOTE inside `ImbPipeline` (no leakage) → XGBoost → Precision-recall threshold tuning → SHAP Beeswarm explainability

**Threshold tuning:** Default 0.5 threshold optimised for business recall — false negatives (missed churners) are more costly than false positives in a retention context.

---

## 🔍 Key Findings (via SHAP Beeswarm)

- **Age** — older customers churn more; SHAP shows clear directional effect
- **Number of products** — customers with only 1 product are significantly higher risk
- **Active membership** — inactive members churn at much higher rates
- **Geography** — German customers show notably higher churn probability
- **Balance** — high balance + inactive = highest risk segment

SHAP Beeswarm plots show not just which features matter but *which direction* they push the prediction — essential for communicating with non-technical stakeholders.

---

## 🛠️ Tech Stack

`Python` `XGBoost` `scikit-learn` `SMOTE` `imbalanced-learn` `ImbPipeline` `SHAP` `matplotlib` `seaborn`

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
jupyter notebook Bank_Customer_Churn_Prediction.ipynb
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `Bank_Customer_Churn_Prediction.ipynb` | Full pipeline: EDA → model → SHAP |

---


