"""Streamlit demo for the trusted frozen churn model."""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import streamlit as st

if not __package__:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

from src.freeze_model import MANIFEST_PATH, FrozenModel, load_frozen_model
from src.input_schema import make_customer_frame
from src.policy import ContactPolicy, contact_decisions
from src.preprocessing import churn_probabilities


def score_customer(frozen: FrozenModel, values: Mapping[str, object]) -> tuple[float, bool]:
    customer = make_customer_frame(frozen.schema, values)
    probabilities = churn_probabilities(frozen.estimator, customer)
    return float(probabilities[0]), bool(contact_decisions(probabilities, frozen.policy)[0])


def format_decision(probability: float, policy: ContactPolicy) -> tuple[str, str | None]:
    if policy.policy_type != "threshold":
        return ("Contact everyone." if policy.policy_type == "contact_all" else "Contact no one."), None
    rule = f"Contact at {policy.threshold:.1%} or higher."
    if f"{probability:.1%}" != f"{policy.threshold:.1%}":
        return rule, None
    if probability == policy.threshold:
        relation = "equal to"
    else:
        relation = "below" if probability < policy.threshold else "above"
    return rule, f"At full precision the probability is {relation} the threshold. The displayed percentages are rounded."


def _widget(field):
    name, label = field["name"], field["label"]
    if "choices" in field:
        if name == "SeniorCitizen":
            return st.selectbox(label, field["choices"], key=name, format_func=lambda value: "Yes" if value else "No")
        return st.selectbox(label, field["choices"], key=name)
    if field["type"] == "integer":
        return st.number_input(label, min_value=int(field["minimum"]), step=1, key=name)
    minimum = float(field["minimum"])
    if field.get("nullable"):
        return st.number_input(label, min_value=minimum, value=None, step=0.01, key=name, placeholder="Unknown")
    return st.number_input(label, min_value=minimum, step=0.01, key=name)


def main() -> None:
    st.set_page_config(page_title="Customer churn estimate", layout="wide")
    st.title("Customer churn estimate")
    st.write("Enter a customer's account and services to estimate churn and apply the frozen outreach policy.")
    manifest = Path(os.getenv("CHURN_FREEZE_MANIFEST", str(MANIFEST_PATH)))
    try:
        frozen = load_frozen_model(manifest, allow_python_patch_difference=True)
    except FileNotFoundError:
        st.error("The frozen model bundle is unavailable. Place the reviewed model, schema, decision file and manifest in the models directory.")
        return
    except (OSError, ValueError) as error:
        st.error(f"The frozen model bundle failed verification. {error}")
        return

    fields = {field["name"]: field for field in frozen.schema["fields"]}
    groups = (
        ("Customer and account", ("gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "Contract", "PaymentMethod", "PaperlessBilling", "MonthlyCharges", "TotalCharges")),
        ("Services", ("PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")),
    )
    with st.form("customer"):
        values = {}
        for columns, (heading, names) in zip(st.columns(2), groups, strict=True):
            with columns:
                st.subheader(heading)
                for name in names:
                    values[name] = _widget(fields[name])
        submitted = st.form_submit_button("Estimate churn", type="primary")

    if submitted:
        try:
            probability, contact = score_customer(frozen, values)
        except ValueError as error:
            st.error(f"Check the customer details. {error}")
            return
        st.metric("Estimated churn probability", f"{probability:.1%}")
        if contact:
            st.success("Contact this customer")
        else:
            st.info("Do not contact this customer")
        rule, detail = format_decision(probability, frozen.policy)
        st.write(rule)
        if detail:
            st.caption(detail)
        st.caption(f"{frozen.decision['freeze_id']} · IBM Telco snapshot estimates")


if __name__ == "__main__":
    main()
