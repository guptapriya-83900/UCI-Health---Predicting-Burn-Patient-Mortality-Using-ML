# 🔥 Burn Mortality Prediction Using Machine Learning

This project uses machine learning to predict in-hospital mortality in burn patients using clinical features such as TBSA, inhalation injury, injury locations, and treatment delays. It outperforms the traditional Baux Score baseline and includes model interpretability and a proposed monitoring framework for deployment.

---

## 📌 Project Highlights

- **Dataset**: Global Burn Registry (5,400+ adult patients)
- **Target**: In-hospital mortality (binary: Survived or Deceased)
- **Model**: Tuned XGBoost (AUC: 0.935)
- **Baseline**: Baux Score logistic regression (AUC: 0.894)
- **Interpretability**: SHAP for local and global explanations
- **Resampling**: ENN, SMOTE, Tomek, and ADASYN
- **Monitoring Plan**: Usage tracking, drift detection, and fairness auditing

---

## 🔍 Problem Statement

Traditional tools like the Baux Score use a linear formula (Age + TBSA + 17 for inhalation injury) to estimate burn mortality. However, clinical reality involves non-linear risk factors and interactions (e.g., injury location, delay in treatment). This project leverages ML to capture these patterns, predict mortality more accurately, and support clinicians with interpretable decisions.

---

## ⚙️ Workflow Overview

1. **Data Cleaning & Preprocessing**
   - Removed missing/invalid outcomes
   - Winsorized outliers in `prehosp_hours`
   - Multi-hot encoded composite fields (e.g., `injury_location`)

2. **Feature Engineering**
   - Injury burden score
   - Age brackets
   - Binary flags for injury sites

3. **Exploratory Data Analysis**
   - Mortality by TBSA, age, injury region, and pre-hospital delay
   - High-risk zones (e.g., perineum/genitals) and features validated

4. **Modeling**
   - Logistic Regression (baseline)
   - Random Forest
   - Tuned XGBoost with `scale_pos_weight`
   - Baux modeled via logistic regression for fair comparison

5. **Imbalance Handling**
   - Compared ENN, SMOTE, Tomek, ADASYN
   - ENN achieved best recall without synthetic data

6. **Interpretability**
   - SHAP summary and waterfall plots
   - Feature-level explanations for individual predictions

7. **Monitoring Plan**
   - Model usage tracking (who, how often)
   - AUC and recall monitoring
   - Data/Concept drift detection
   - Fairness analysis by sex, age, TBSA groups

---

## 📈 Results

| Model        | AUC   | Recall (Dead) | F1 Score (Dead) |
|--------------|-------|----------------|------------------|
| **XGBoost**   | 0.935 | 0.89           | 0.75             |
| Baux Score   | 0.894 | 0.66           | 0.67             |
| Logistic Reg | 0.91  | 0.84           | 0.73             |

> ✅ ML captured patterns Baux missed — like survival despite high TBSA due to short delays and early surgery.

---

## 🧠 Tools Used

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, imbalanced-learn
- SHAP for interpretability
- GridSearchCV for tuning

---

## 🏥 Future Work

- Integration with hospital EHR for real-time use
- Prospective validation with new data
- Deployment using Flask + Docker
- Integration with monitoring tools like EvidentlyAI

---

## 📄 License

This project is for academic and research use only. Attribution required for dataset use (Global Burn Registry guidelines).


