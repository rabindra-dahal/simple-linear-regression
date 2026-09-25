# app.py
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

# 1. Page Configuration Settings
st.set_page_config(
    page_title="Swedish Auto Insurance Dashboard",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🚗 Swedish Auto Insurance Dashboard")
# FIXED: Prefixing with 'r' converts this into a raw string to protect backslashes
st.write(
    r"**Architecture Platform:** Decoupled Architecture (`Streamlit Client` → `FastAPI Backend Service`)"
)

# Network Configuration Coordinates Mapping
API_URL = "http://127.0.0.1:8000"

# 2. Sidebar Integration & System Health Dependency Tracker
st.sidebar.header("System Connectivity Status")


def check_api_health():
    """Queries the backend health metrics to confirm server availability and operational modes."""
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            return health_resp.json()
    except Exception:
        return None
    return None


health_data = check_api_health()

if health_data is None:
    st.sidebar.error("🔴 API Backend: Offline")
    st.sidebar.write("Ensure your FastAPI microservice is executing on port 8000.")
else:
    status_state = health_data.get("status")
    deps = health_data.get("dependencies", {})

    if status_state == "healthy":
        st.sidebar.success("🟢 API Backend: Live & Online")
        st.sidebar.caption("Data Source: Remote GitHub Repository Matrix")
    elif status_state == "degraded":
        st.sidebar.warning("🟡 API Backend: Degraded (Offline Mode)")
        st.sidebar.caption(
            "Data Source: Running seamlessly via local backup file fallback."
        )
    else:
        st.sidebar.error("❌ API Backend: Critical Data Missing")

# Sidebar Parameter Control Sliders
st.sidebar.markdown("---")
st.sidebar.header("Model Pipeline Configuration")
test_split = st.sidebar.slider(
    "Test Split Ratio",
    min_value=0.01,  # Lower bound threshold setting for validation checks
    max_value=1.2,  # Set above 0.95 to deliberately trigger out-of-bounds 422 validations
    value=0.2,
    step=0.05,
    help="Adjust partition ratios to validate data bounds constraints.",
)


# 3. Model Engine Pipeline Processing Fetch Implementation
def fetch_model_data(split):
    """Pings endpoint rules to optimize models, returning training matrices or parsed exceptions."""
    try:
        response = requests.post(
            f"{API_URL}/train-and-metrics", json={"test_size": split}
        )

        if response.status_code == 200:
            return response.json(), None

        elif response.status_code == 422:
            # Parse schema exception array structures sent back from Pydantic fields
            err_detail = response.json().get(
                "details", [{"msg": "Input criteria mismatch."}]
            )
            parsed_errors = "\n".join(
                [f"- {err['msg']}" for err in err_detail]
            )
            return (
                None,
                f"**Validation Error (422):** The server rejected the parameters:\n{parsed_errors}",
            )

        elif response.status_code == 404:
            detail_msg = response.json().get(
                "detail", "The target data asset could not be accessed."
            )
            return None, f"**Asset Error (404):** {detail_msg}"

        else:
            return (
                None,
                f"**Server Error ({response.status_code}):** An unexpected runtime pipeline failure occurred.",
            )

    except requests.exceptions.ConnectionError:
        return (
            None,
            "**Connection Error:** Could not contact the backend microservice. Verify your server script status.",
        )


# Run dashboard population execution routine
data_payload, error_msg = fetch_model_data(test_split)

if error_msg:
    st.error(error_msg)
else:
    # 4. Render KPI Metrics Boards
    col1, col2, col3 = st.columns(3)
    col1.metric(label="R² Score", value=f"{data_payload['r2']:.4f}")
    col2.metric(label="RMSE Score", value=f"{data_payload['rmse']:.2f}")
    col3.metric(label="MSE Score", value=f"{data_payload['mse']:.2f}")

    # FIXED: Prefixing with 'fr' enables string formatting while handling backslashes cleanly as raw text
    st.info(
        fr"**Model Regression Formula:**  \(Y = {data_payload['slope']:.2f}X + {data_payload['intercept']:.2f}\)"
    )

    # 5. Visual Render Plots Framework
    raw_claims = data_payload["raw_data"]["Claims"]
    raw_payment = data_payload["raw_data"]["Payment"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(raw_claims, raw_payment, color="darkblue", label="Source Data")

    # Draw fit-line boundaries systematically based on core metrics parameters returned
    x_line = np.linspace(min(raw_claims), max(raw_claims), 100)
    y_line = data_payload["slope"] * x_line + data_payload["intercept"]

    ax.plot(
        x_line,
        y_line,
        color="crimson",
        linewidth=2.5,
        label="API Computed Best Fit Line",
    )
    ax.set_xlabel("Number of Claims")
    ax.set_ylabel("Total Payment (Thousands of Kronor)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # 6. Remote Real-Time Inference Section
    st.subheader("🔮 Remote Inference Engine")

    # Min-value bound configured lower to demonstrate validation exception workflows visually
    user_claims = st.number_input(
        "Enter Number of Claims:", min_value=-50, value=20, step=1
    )

    if st.button("Calculate Payment Target"):
        try:
            pred_response = requests.post(
                f"{API_URL}/predict",
                json={"claims": user_claims, "test_size": test_split},
            )

            if pred_response.status_code == 200:
                pred_res = pred_response.json()
                st.success(
                    f"Estimated total payment: **{pred_res['predicted_payment']:.2f} thousand Kronor**."
                )
            elif pred_response.status_code == 422:
                err_detail = pred_response.json().get(
                    "details", [{"msg": "Input verification failure."}]
                )
                parsed_errors = "\n".join(
                    [f"- {err['msg']}" for err in err_detail]
                )
                st.warning(
                    f"**Inference Blocked (422):** Invalid attributes:\n{parsed_errors}"
                )
            else:
                st.error(
                    f"Inference request failed with status code: {pred_response.status_code}."
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Inference pipeline calculation dropped because the server dropped offline."
            )
