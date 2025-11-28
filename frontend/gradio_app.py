import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx
import pandas as pd
from loguru import logger

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import load_config

# Constants
FEATURE_ORDER: List[str] = [
    "Age",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Male",
    "Age_Group",
    "Interaction_Frequency",
]
TIMEOUT_SECONDS: float = 10.0


def predict_churn(
    customer_id: float,
) -> Tuple[str, float, str, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Orchestrates the prediction flow: validates input, queries the API,
    and formats the response for the UI.

    Args:
        customer_id: The unique identifier for the customer (input from UI).

    Returns:
        A tuple containing:
        - str: The prediction label ("CHURN", "NO CHURN", or "Error").
        - float: The probability score (0.0 to 1.0).
        - str: The model version used.
        - Optional[pd.DataFrame]: A DataFrame of the features used for inference.
        - Dict[str, Any]: The raw API response or an error dictionary.

    Raises:
        None: All exceptions are caught and returned as error states for the UI.
    """
    # 1. Validation
    if customer_id is None:
        return (
            "Error",
            0.0,
            "N/A",
            None,
            {"error": "Please enter a Customer ID."},
        )

    if customer_id <= 0:
        return (
            "Error",
            0.0,
            "N/A",
            None,
            {"error": "Customer ID must be a positive number."},
        )

    try:
        config = load_config()
        if not config.fastapi_url:
            raise ValueError("FASTAPI_URL is not set in configuration.")

        api_url = f"{config.fastapi_url}/predict/{int(customer_id)}"
    except Exception as e:
        return "Error", 0.0, "N/A", None, {"error": f"Configuration Failure: {e}"}

    # 2. API Interaction
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get(api_url)
            response.raise_for_status()
            data = response.json()

        # 3. Response Parsing
        prediction_val = data.get("prediction")
        probability_val = data.get("probability", 0.0)
        version_val = data.get("version", "Unknown")
        features_val = data.get("features", {})

        if prediction_val is None:
            raise KeyError("API response missing 'prediction' key.")

        # 4. Formatting
        label = "CHURN" if prediction_val == 1 else "NO CHURN"

        # Create DataFrame and enforce column order
        features_df = pd.DataFrame([features_val])
        features_df = features_df.reindex(columns=FEATURE_ORDER).fillna(0)

        return label, probability_val, version_val, features_df, data

    except httpx.ConnectError:
        err_msg = "Connection Refused: Is the FastAPI backend running?"
        return "Error", 0.0, "N/A", None, {"error": err_msg}

    except httpx.HTTPStatusError as e:
        err_msg = f"API Error {e.response.status_code}: {e.response.text}"
        return "Error", 0.0, "N/A", None, {"error": err_msg}

    except (ValueError, KeyError, Exception) as e:
        return "Error", 0.0, "N/A", None, {"error": f"Processing Error: {str(e)}"}


def load_data(table_choice: str) -> Tuple[pd.DataFrame, str]:
    """
    Fetches recent data rows from the backend API for a specific table.

    Args:
        table_choice: The user-selected table name (e.g., "Prediction Logs").

    Returns:
        A tuple containing:
        - pd.DataFrame: The retrieved data. Returns empty DataFrame on error.
        - str: A status message indicating success or failure.

    Raises:
        None: Exceptions are handled and returned as status messages.
    """
    table_map: Dict[str, str] = {
        "Prediction Logs": "predictions",
        "Model Performance": "performance",
        "Customers": "customers",
    }

    endpoint_suffix = table_map.get(table_choice)
    if not endpoint_suffix:
        return pd.DataFrame(), "Error: Invalid table selection."

    try:
        config = load_config()
        if not config.fastapi_url:
            return pd.DataFrame(), "Error: FASTAPI_URL config missing."

        url = f"{config.fastapi_url}/data/{endpoint_suffix}"

        # Increased timeout for data fetching
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

        if not data:
            return pd.DataFrame(), f"No data returned for '{table_choice}'."

        df = pd.DataFrame(data)
        return df, f"Successfully loaded {len(df)} rows."

    except httpx.ConnectError:
        return pd.DataFrame(), "Error: Could not connect to backend."
    except httpx.HTTPStatusError as e:
        return pd.DataFrame(), f"API Error: {e.response.status_code}"
    except Exception as e:
        return pd.DataFrame(), f"Unexpected Error: {e}"


# --- UI Construction ---
with gr.Blocks(
    theme=gr.themes.Soft(),
    js="""
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}""",
) as demo:
    gr.Markdown("# 📉 Customer Churn Prediction Dashboard")

    with gr.Tab("Live Prediction"):
        gr.Markdown("### Real-time Inference")
        gr.Markdown(
            "Enter a **Customer ID** to retrieve features from the Feature Store "
            "and generate a prediction."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_customer_id = gr.Number(
                    label="Customer ID", value=1001, precision=0, minimum=1
                )
                btn_predict = gr.Button("Predict Churn", variant="primary")

            with gr.Column(scale=2):
                with gr.Group():
                    out_label = gr.Label(label="Prediction Result")
                    out_prob = gr.Slider(
                        label="Churn Probability", minimum=0, maximum=1
                    )

                with gr.Accordion("Model Details & Features", open=True):
                    out_version = gr.Textbox(label="Model Version")
                    out_features = gr.Dataframe(
                        label="Input Features", type="pandas", interactive=False
                    )

                with gr.Accordion("Raw API Response", open=False):
                    out_json = gr.JSON(label="Debug Info")

        btn_predict.click(
            fn=predict_churn,
            inputs=[input_customer_id],
            outputs=[out_label, out_prob, out_version, out_features, out_json],
        )

    with gr.Tab("Data Explorer"):
        gr.Markdown("### Production Database View")
        with gr.Row():
            input_table = gr.Dropdown(
                choices=["Prediction Logs", "Model Performance", "Customers"],
                value="Prediction Logs",
                label="Select Table",
            )
            btn_load = gr.Button("Load Data", variant="secondary")

        out_status = gr.Textbox(label="Status", interactive=False)
        out_data = gr.Dataframe(
            label="Table Contents", interactive=False, max_height=500
        )

        btn_load.click(
            fn=load_data,
            inputs=[input_table],
            outputs=[out_data, out_status],
        )

if __name__ == "__main__":
    try:
        logger.info("Starting Gradio Interface...")
        demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
    except OSError as e:
        logger.error(f"Failed to launch on port 7860: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Stopping Gradio Interface...")
        sys.exit(0)
