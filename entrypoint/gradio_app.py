import gradio as gr
import httpx
import pandas as pd
import os

# Get the FastAPI URL from an environment variable
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi_server:8000")

# Define the order of features to display
FEATURE_ORDER = [
    "Age", "Support Calls", "Payment Delay", "Total Spend", 
    "Last Interaction", "Male", "Age_Group", "Interaction_Frequency"
]

def predict_churn(customer_id):
    """
    Calls the FastAPI service to get a churn prediction and the features used.
    """
    if customer_id is None:
        return "No prediction", 0.0, "N/A", None, "Please enter a Customer ID."

    try:
        # 1. Call your existing FastAPI endpoint
        response = httpx.get(f"{FASTAPI_URL}/predict/{int(customer_id)}")
        
        # Handle API errors
        if response.status_code != 200:
            error_msg = response.json().get('detail', 'Unknown API error')
            return "Error", 0.0, "N/A", None, f"API Error: {error_msg}"
        
        data = response.json()
        
        # 2. Format the outputs for Gradio components
        prediction = data['prediction']
        probability = data['probability']
        model_version = data['version']
        features = data['features']
        
        # Format for gr.Label
        prediction_label = "CHURN" if prediction == 1 else "NO CHURN"
        
        # Format for gr.Dataframe
        features_df = pd.DataFrame([features], columns=FEATURE_ORDER)
        
        # Format for gr.JSON
        raw_json_response = data
        
        return prediction_label, probability, model_version, features_df, raw_json_response

    except httpx.ConnectError:
        return "Error", 0.0, "N/A", None, "Connection Error: Could not connect to the FastAPI service."
    except Exception as e:
        return "Error", 0.0, "N/A", None, f"An error occurred: {str(e)}"

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
                label="Customer ID", 
                value=1001, 
                precision=0
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
                        label="Model Version", 
                        interactive=False
                    )

                # --- Tab 2: Features Used ---
                with gr.Tab("Features Used"):
                    gr.Markdown("These features were fetched *in real-time* from the online store to make the prediction.")
                    features_output = gr.Dataframe(
                        label="Online Features",
                        headers=FEATURE_ORDER,
                        datatype=["number"] * len(FEATURE_ORDER)
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
    demo.launch(server_name="0.0.0.0", server_port=7860)