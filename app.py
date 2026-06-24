"""Streamlit front end for the Telco churn retention workflow."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "churn_model.pkl"
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@st.cache_resource
def load_artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python train_model.py` first."
        )
    # Only load the artifact generated locally by train_model.py; pickle files
    # from untrusted sources can execute code when deserialized.
    with path.open("rb") as model_file:
        artifact = pickle.load(model_file)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary created by train_model.py.")
    required_keys = {"model", "feature_columns", "threshold"}
    if not required_keys.issubset(artifact):
        raise ValueError(f"Model artifact is missing keys: {required_keys - artifact.keys()}")
    return artifact


def has_high_risk_pattern(customer: pd.DataFrame | pd.Series) -> bool:
    row = customer.iloc[0] if isinstance(customer, pd.DataFrame) else customer
    return bool(
        row["Contract"] == "Month-to-month"
        and float(row["MonthlyCharges"]) >= 80
        and row["TechSupport"] == "No"
    )


def retention_action(customer: pd.DataFrame, risk: float, threshold: float) -> str:
    row = customer.iloc[0]
    if risk >= threshold and row["Contract"] == "Month-to-month":
        return (
            "Offer a 15% discount for the first 3 months when upgrading to a "
            "one-year contract; confirm the customer values the included services."
        )
    if risk >= threshold and row["TechSupport"] == "No":
        return "Offer a 3-month complimentary Tech Support trial and schedule a follow-up."
    if risk >= 0.4:
        return "Schedule a proactive account check-in and review recent service issues."
    return "No immediate save offer. Keep the account in the standard monitoring queue."


def prepare_customer(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required customer fields: {', '.join(sorted(missing))}")
    customer = frame.loc[:, columns].copy()
    customer["TotalCharges"] = pd.to_numeric(customer["TotalCharges"], errors="coerce")
    customer["MonthlyCharges"] = pd.to_numeric(customer["MonthlyCharges"], errors="coerce")
    customer["tenure"] = pd.to_numeric(customer["tenure"], errors="coerce")
    if customer[["MonthlyCharges", "tenure"]].isna().any().any():
        raise ValueError("MonthlyCharges and tenure must be valid numeric values.")
    return customer


def customer_form() -> tuple[pd.DataFrame, bool]:
    with st.form("customer_profile"):
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
            partner = st.selectbox("Has a partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=1.0)
            total_charges_text = st.text_input(
                "Total charges ($)",
                value="",
                help="Leave blank for a brand-new customer; the model handles this as missing.",
            )
        with right:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        submitted = st.form_submit_button("Assess churn risk")

    total_charges = pd.to_numeric(pd.Series([total_charges_text]), errors="coerce").iloc[0]
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )
    return customer, submitted


def score_customer(customer: pd.DataFrame, artifact: dict) -> tuple[float, bool]:
    rule_triggered = has_high_risk_pattern(customer.iloc[0])
    if rule_triggered:
        # The transparent baseline is intentionally checked first. We still
        # score with the model, but never let it silently contradict this flag.
        st.warning(
            "Rule-based early flag: month-to-month contract, monthly charges "
            "of at least $80, and no Tech Support."
        )
    probability = float(artifact["model"].predict_proba(customer)[0, 1])
    if rule_triggered:
        probability = max(probability, 0.75)
    return probability, rule_triggered


def main() -> None:
    st.set_page_config(page_title="Telco Retention Desk", page_icon="📡", layout="wide")
    st.title("Telco Retention Desk")
    st.caption("A decision-support prototype: risk signals should inform, not replace, an account conversation.")
    try:
        artifact = load_artifact(MODEL_PATH)
    except (FileNotFoundError, ValueError, OSError, pickle.UnpicklingError) as error:
        st.error(str(error))
        st.stop()

    operations_tab, leadership_tab = st.tabs(["Operations · Account review", "Leadership · Portfolio review"])
    with operations_tab:
        st.subheader("Assess one customer")
        customer, submitted = customer_form()
        if submitted:
            try:
                customer = prepare_customer(customer, artifact["feature_columns"])
                risk, rule_triggered = score_customer(customer, artifact)
            except ValueError as error:
                st.error(str(error))
            else:
                st.metric("Estimated churn risk", f"{risk:.0%}")
                if risk >= artifact["threshold"] or rule_triggered:
                    st.error("Priority follow-up recommended")
                elif risk >= 0.4:
                    st.warning("Monitor and check in proactively")
                else:
                    st.success("No immediate escalation indicated")
                st.markdown("#### Retention Playbook Action")
                st.info(retention_action(customer, risk, artifact["threshold"]))
                st.caption(
                    "The rule-based flag is a transparent safety net, not a causal "
                    "claim. Validate offers with customer preferences and policy."
                )

    with leadership_tab:
        st.subheader("Score a customer portfolio")
        st.write(
            "Upload a customer CSV with the same feature columns as the supplied Telco data. "
            "The export adds risk, a rule flag, and a revenue-at-risk estimate."
        )
        uploaded = st.file_uploader("Customer CSV", type=["csv"], key="portfolio_upload")
        if uploaded is not None:
            try:
                portfolio = pd.read_csv(uploaded)
                customer_features = prepare_customer(portfolio, artifact["feature_columns"])
                probabilities = artifact["model"].predict_proba(customer_features)[:, 1]
                rule_flags = customer_features.apply(has_high_risk_pattern, axis=1)
                adjusted = pd.Series(probabilities, index=customer_features.index)
                adjusted.loc[rule_flags] = adjusted.loc[rule_flags].clip(lower=0.75)
                result = portfolio.copy()
                result["churn_risk"] = adjusted
                result["rule_based_flag"] = rule_flags
                result["annual_revenue_at_risk"] = (
                    pd.to_numeric(customer_features["MonthlyCharges"], errors="coerce")
                    * 12
                    * adjusted
                )
                result["retention_action"] = [
                    retention_action(
                        customer_features.iloc[[position]],
                        float(risk),
                        artifact["threshold"],
                    )
                    for position, risk in enumerate(adjusted)
                ]
                st.metric("Customers in upload", len(result))
                st.metric(
                    "Expected annual revenue exposure",
                    f"${result['annual_revenue_at_risk'].sum():,.0f}",
                )
                st.dataframe(result.head(100), use_container_width=True)
                st.download_button(
                    "Download scored portfolio",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="telco_churn_scored.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Revenue exposure is a simple 12-month estimate "
                    "(monthly charges × 12 × churn probability), not a forecast of realized loss."
                )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not score this upload: {error}")


if __name__ == "__main__":
    main()
