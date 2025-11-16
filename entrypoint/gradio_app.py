import os
import sys

# Ensures the project root is in the Python path for core imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from typing import Any, List, Optional, Tuple

import gradio as gr
import httpx
import pandas as pd

from core.config import load_config

# Define the order of features to display
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


def predict_churn(
    customer_id: Optional[float],
) -> Tuple[str, float, str, Optional[pd.DataFrame], Any]:
    """
    Calls the FastAPI service to get a churn prediction and the features used.

    The function handles input validation, API connection errors, HTTP status
    errors, and response parsing errors (JSON, KeyError). All errors are
    caught and returned as a valid tuple for the Gradio UI.

    Args:
        customer_id (Optional[float]): The customer ID from the Gradio input.
            It can be a float (from gr.Number) or None.

    Returns:
        Tuple[str, float, str, Optional[pd.DataFrame], Any]: A 5-tuple
        mapping directly to the Gradio outputs:
        - (str) Prediction label ("CHURN" / "NO CHURN" / "Error").
        - (float) Churn probability (0.0 to 1.0), 0.0 on error.
        - (str) Model version string, "N/A" on error.
        - (Optional[pd.DataFrame]) DataFrame of features, or None on error.
        - (Any) The raw JSON API response dict, or an error message string.
    """
    # 1. Load config (inside function for robustness)
    try:
        config = load_config()
    except Exception as e:
        error_msg = f"Configuration Error: Failed to load config. {e}"
        return "Error", 0.0, "N/A", None, error_msg

    # 2. Validate input
    if customer_id is None or customer_id <= 0:
        return (
            "No prediction",
            0.0,
            "N/A",
            None,
            "Please enter a valid, positive Customer ID.",
        )

    try:
        customer_id_int = int(customer_id)
    except ValueError:
        return "Error", 0.0, "N/A", None, "Invalid Customer ID: Must be a whole number."

    # 3. Call API and handle all errors
    try:
        api_url = f"{config.fastapi_url}/predict/{customer_id_int}"

        # Use httpx.Client for proper timeout and resource management
        with httpx.Client(timeout=10.0) as client:
            response = client.get(api_url)

            # Raise exceptions for 4xx/5xx responses
            response.raise_for_status()

        # 4. Process successful response
        data = response.json()  # Can raise json.JSONDecodeError

        # 5. Extract and validate keys
        prediction = data["prediction"]  # Can raise KeyError
        probability = data["probability"]
        model_version = data["version"]
        features = data["features"]

        # 6. Format outputs
        prediction_label = "CHURN" if prediction == 1 else "NO CHURN"

        # Robust DataFrame creation: create from dict, then reindex
        # This prevents errors if API omits a feature or changes order
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=FEATURE_ORDER)

        return (
            prediction_label,
            probability,
            model_version,
            features_df,
            data,  # Return the raw JSON dict
        )

    # --- Specific Error Handlers ---
    except httpx.ConnectError as e:
        error_msg = f"Connection Error: Could not connect to the API at {config.fastapi_url}. {e}"
        return "Error", 0.0, "N/A", None, error_msg

    except httpx.HTTPStatusError as e:
        # Try to parse error detail from API, fall back to status code
        try:
            error_detail = e.response.json().get("detail", e.strerror)
        except json.JSONDecodeError:
            error_detail = e.response.text or e.strerror
        error_msg = f"API Error ({e.response.status_code}): {error_detail}"
        return "Error", 0.0, "N/A", None, error_msg

    except httpx.RequestError as e:
        # Catch other request errors (e.g., ReadTimeout)
        error_msg = f"API Request Error: {e}"
        return "Error", 0.0, "N/A", None, error_msg

    except json.JSONDecodeError:
        error_msg = (
            "API Error: Failed to decode successful (200) JSON response from server."
        )
        return "Error", 0.0, "N/A", None, error_msg

    except KeyError as e:
        error_msg = f"API Error: Response missing expected key: {e}. Check API schema."
        return "Error", 0.0, "N/A", None, error_msg

    except Exception as e:
        # Generic fallback for any other unexpected error
        error_msg = f"An unexpected error occurred: {str(e)}"
        return "Error", 0.0, "N/A", None, error_msg


# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        #  Customer Churn Prediction
        Enter a Customer ID to fetch real-time features from the Feast online store
        and get a live prediction from the deployed model.
        """
    )

    with gr.Row():
        # --- INPUTS ---
        with gr.Column(scale=1):
            customer_id_input = gr.Number(
                label="Customer ID", value=1001, precision=0  # Requires whole numbers
            )
            predict_button = gr.Button("Predict", variant="primary")

        # --- OUTPUTS ---
        with gr.Column(scale=3):
            with gr.Tabs():
                # --- Tab 1: Prediction Result (Default) ---
                with gr.Tab("Prediction Result"):
                    prediction_output = gr.Label(label="Prediction")
                    probability_output = gr.Slider(
                        label="Churn Probability",
                        minimum=0.0,
                        maximum=1.0,
                    )
                    version_output = gr.Textbox(
                        label="Model Version", interactive=False
                    )

                # --- Tab 2: Features Used ---
                with gr.Tab("Features Used"):
                    gr.Markdown(
                        "These features were fetched *in real-time* from the online store to make the prediction."
                    )
                    features_output = gr.Dataframe(
                        label="Online Features",
                        headers=FEATURE_ORDER,
                        datatype=["number"] * len(FEATURE_ORDER),
                    )

                # --- Tab 3: Raw API Response ---
                with gr.Tab("Raw API Response"):
                    json_output = gr.JSON(label="API Response")

    # Connect the button to the function and components
    predict_button.click(
        fn=predict_churn,
        inputs=[customer_id_input],
        outputs=[
            prediction_output,
            probability_output,
            version_output,
            features_output,
            json_output,
        ],
    )

if __name__ == "__main__":
    try:
        print("Launching Gradio app on http://0.0.0.0:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        # Catch errors like "Address already in use"
        print(f"Failed to launch Gradio app: {e}", file=sys.stderr)
        sys.exit(1)
