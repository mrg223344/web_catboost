# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set up the page configuration
st.set_page_config(
    page_title="POI Prediction Platform",
    page_icon="🩺",
    layout="wide"
)

# 1. Load Model and SHAP Explainer
@st.cache_resource
def load_model_and_explainer():
    """
    Loads the saved CatBoost model and creates the SHAP explainer.
    Caches the results for performance.
    """
    try:
        # The filename 'catboost_poi_model.joblib' must exist in the same directory
        model = joblib.load('catboost_poi_model.joblib')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        st.error("ERROR: Model file 'catboost_poi_model.joblib' not found.")
        st.error("Please run 'train_model.py' first to generate the model file.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, explainer = load_model_and_explainer()

# Stop execution if model loading failed
if model is None:
    st.stop()

# 2. Streamlit UI
st.title("👨‍⚕️ Postoperative Ileus (POI) Prediction Platform ")
st.markdown("Forecasting and Interpretability Analysis using **CatBoost** and **SHAP**")

# 3. Sidebar for User Input
st.sidebar.header("Patient Data Input:")

# Feature: Surgery Duration (min)
surgery_time = st.sidebar.number_input(
    'Surgery Duration (min)', 
    min_value=0, max_value=1000, value=120,
    help="Enter the total time of the surgical procedure in minutes."
)

# Feature: Levels of Surgery (Binary 0/1) - MODIFIED to binary
levels = st.sidebar.selectbox(
    'Levels of Surgery (>= 3 levels = 1)', 
    options=[0, 1], # Binary options
    index=1,
    help="1 = 3 or more surgical segments/levels involved. 0 = Less than 3." # Updated help text
)

# Feature: Preoperative BUN (mmol/L)
pre_bun = st.sidebar.number_input(
    'Preoperative BUN (mmol/L)', 
    min_value=0.0, max_value=50.0, value=5.8, format="%.2f",
    help="Preoperative Blood Urea Nitrogen level in mmol/L."
)

# Feature: ASA Classification (Binary 0/1) - MODIFIED to binary
asa = st.sidebar.selectbox(
    'ASA Classification (>= 3 = 1)', 
    options=[0, 1], # Binary options
    index=1,
    help="1 = ASA Physical Status Classification score of 3 or higher. 0 = ASA score less than 3." # Updated help text
)

# Feature: Postoperative 24h NRS (Binary 0/1)
nrs = st.sidebar.selectbox(
    'Postoperative 24h NRS (>= 3 points = 1)', 
    options=[0, 1], index=0,
    help="Numeric Rating Scale score at 24h post-op (0 = < 3, 1 = >= 3)."
)

# Feature: Preoperative Abdominal Distention or Constipation (Binary 0/1)
pre_disten_consti = st.sidebar.selectbox(
    'Preoperative Distention/Constipation', 
    options=[0, 1], 
    index=0,
    help="1 = Present, 0 = Absent."
)

# 4. Prepare Input Data
# CRITICAL: Column names and order must match the training set exactly
features_columns = ['surgery_time', 'levels', 'pre_BUN', 'ASA', 'NRS', 'pre_disten_consti']

input_data = pd.DataFrame(
    [[surgery_time, levels, pre_bun, asa, nrs, pre_disten_consti]],
    columns=features_columns
)

# Convert categorical features back to 'category' dtype as used in training
# NOTE: All binary features (levels, ASA, NRS, pre_disten_consti) are treated as categorical.
input_data['levels'] = input_data['levels'].astype('category') # ADDED
input_data['ASA'] = input_data['ASA'].astype('category')
input_data['NRS'] = input_data['NRS'].astype('category')
input_data['pre_disten_consti'] = input_data['pre_disten_consti'].astype('category')

st.subheader("1. Current Input Data")
st.dataframe(input_data, use_container_width=True)

# 5. Make Prediction
if st.sidebar.button("Predict POI"):
    # Predict probability for class 1 (POI = 1)
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("2. Prediction Result")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Class (POI)", value=f"{prediction}", help="0 = Predicted NO POI, 1 = Predicted POI")
    
    with col2:
        # Dynamic color based on risk level
        delta_color = "inverse" if prediction_proba < 0.5 else "normal"
        st.metric(
            label="Probability of POI",
            value=f"{prediction_proba * 100:.2f} %",
            delta=f"{(prediction_proba - 0.5)*100:.2f} % vs 50% Threshold",
            delta_color=delta_color
        )

    # 6. SHAP Interpretability Analysis
    st.subheader("3. Prediction Explainability (SHAP Analysis)")
    st.markdown("""
    The SHAP Force Plot below shows how each feature pushes the prediction from the "base value" (the average prediction probability of the model) to the final prediction probability.
    - <span style='color:red;'>**Red features**</span> increase the probability of POI.
    - <span style='color:blue;'>**Blue features**</span> decrease the probability of POI.
    - The size of the feature label indicates the magnitude of its contribution.
    """, unsafe_allow_html=True)
    
    try:
        # Calculate SHAP values for class 1 (POI)
        shap_values = explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            # For binary classification, use the values for class 1
            shap_values_for_plot = shap_values[1]
            expected_value_for_plot = explainer.expected_value[1]
        else:
            # For single output (regression or single-class)
            shap_values_for_plot = shap_values
            expected_value_for_plot = explainer.expected_value

        # Plot the SHAP Force Plot
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            base_value=expected_value_for_plot,
            shap_values=shap_values_for_plot,
            features=input_data,
            matplotlib=True,
            show=False
        )
        # Display the matplotlib figure in Streamlit
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.close() # Close figure to prevent memory leak

    except Exception as e:
        st.error(f"Error calculating or plotting SHAP: {e}")

else:
    st.info("Please enter the patient data in the sidebar and click 'Predict POI'.")
