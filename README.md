# Telco Churn Retention Desk

A deployed Streamlit decision-support app built around the Telco Customer Churn dataset. It combines a scikit-learn logistic-regression pipeline, a transparent high-risk rule, and practical retention recommendations.

## Portfolio description

> Estimate whether a Telco customer is likely to churn and provide risk flags and practical retention recommendations.

`Python` `Streamlit` `Scikit-learn` `Pandas` `Render` — *End-to-end machine learning workflow and interactive decision-support app*

**Live demo:** [Open the Telco Churn Retention Desk](https://telco-churn-retention-desk.onrender.com)


## Run locally

From this folder, install dependencies and train the model:

```powershell
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` prints validation ROC AUC, PR AUC, and Brier score, then writes `churn_model.pkl`. The app will show an explicit setup message if that artifact has not been created yet. Keep the artifact local or deploy only a model you trained and trust: Python pickle files are executable during loading.

## Data-cleaning decision

The source `TotalCharges` field is text and contains blank strings for some brand-new customers with zero months of tenure. Training and inference both use `pandas.to_numeric(..., errors="coerce")`; blanks become missing values, and the fitted numeric pipeline imputes them. The application allows the total-charge field to be left blank instead of failing on that valid customer state.

## App workflow

- **Operations · Account review:** enter a customer profile, review its risk estimate, and get a retention action.
- **Leadership · Portfolio review:** upload a CSV with the Telco feature columns and download risk, rule flags, action suggestions, and estimated annual revenue exposure.
- **Rule-based early flag:** month-to-month contract + monthly charges of at least $80 + no Tech Support is surfaced before model scoring. The score is floored at 75% when this transparent rule fires; it is a conservative escalation heuristic, not a causal claim.
- **Priority cutoff:** a 65% estimated churn risk is the demo's business threshold for a high-risk action; change it only with an explicit retention-capacity/economics rationale.
- **Revenue at risk:** estimated as monthly charges × 12 × churn probability. This is a simple planning proxy, not a realized-loss forecast.

The discount and support-trial recommendations are examples for a portfolio demo; a real retention team should validate eligibility, economics, and customer preferences before making offers.
