# Description: This code sets up a Streamlit user interface for stroke prediction using a pre-trained model.
"""
1. Import the necessary libraries: `streamlit`, `pandas`, `numpy`, and `joblib`.
2. Load the pre-trained model and preprocessors using `joblib.load()`.
3. Set up the Streamlit user interface with a title.
4. Create input fields for user data: age, weight, height, hypertension, heart disease, ever married, and work type.
5. Add a checkbox for unknown glucose level.
6. Display the input field for average glucose level if the checkbox is not checked.
7. When the "Predict Stroke Risk" button is clicked:
   - Create a DataFrame with the user input data.
   - Handle unknown glucose level by setting it to NaN if the checkbox is checked.
   - Apply preprocessing steps to the input data using the loaded preprocessors.
   - Make the prediction using the pre-trained model.
   - Display the prediction result: high risk or low risk of stroke.

This code provides a user-friendly interface for users to input their data and receive a stroke risk prediction based on the pre-trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and preprocessors
model, scaler, label_encoder, imputer = joblib.load("stroke_model_pipeline.joblib")

# Set up the Streamlit user interface for stroke prediction
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠")

# Add custom CSS styles
st.markdown(
    """
    <style>
    .header {
        font-size: 40px;
        font-weight: bold;
        color: #3498db;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #3498db;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
    }
    .error {
        color: #e74c3c;
    }
    .success {
        color: #2ecc71;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">Stroke Prediction App</div>', unsafe_allow_html=True)
st.write("This app predicts your risk of stroke based on the provided information.")

# User input section
st.markdown('<div class="subheader">Enter your details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, format="%.2f")
    height = st.number_input(
        "Height (meters)", min_value=0.5, max_value=2.5, value=1.75
    )
    bmi = weight / (height**2)
    st.write(f"BMI: {bmi:.2f}")

with col2:
    col2_1, col2_2, col2_3 = st.columns(3)
    
    with col2_1:
        hypertension = st.radio("Hypertension", ["No", "Yes"])
    
    with col2_2:
        heart_disease = st.radio("Heart disease", ["No", "Yes"])
    
    with col2_3:    
        ever_married = st.radio("Ever married", ["No", "Yes"])
    
    work_type = st.selectbox(
        "Work type",
        ["Government job", "Never worked", "Private", "Self-employed", "Children"],
    )
    
    # Add a checkbox for unknown glucose level
    unknown_glucose = st.checkbox("Glucose level unknown")

    # Display the input field for average glucose level if the checkbox is not checked
    if not unknown_glucose:
        avg_glucose_level = st.number_input(
            "Average Glucose Level (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=80.0,
            help="A normal glucose level is between 70 to 99 mg/dL",
        )
    else:
        avg_glucose_level = np.nan

# Add an expander with additional information
with st.expander("Learn more about stroke risk factors"):
    st.write(
        """
        - Age: The risk of stroke increases with age.
        - Hypertension: High blood pressure is a major risk factor for stroke.
        - Heart disease: Conditions such as coronary artery disease and atrial fibrillation can increase the risk of stroke.
        - Diabetes: High blood sugar levels can damage blood vessels and increase the risk of stroke.
        - Obesity: Being overweight or obese increases the risk of stroke.
        - Lifestyle factors: Smoking, physical inactivity, and unhealthy diet can contribute to stroke risk.
        """
    )

# Perform prediction when the user clicks the "Predict Stroke Risk" button
if st.button("Predict Stroke Risk"):
    # Create a DataFrame with the user input data
    input_data = pd.DataFrame(
        [
            [
                age,
                (1 if hypertension == "Yes" else 0),
                avg_glucose_level,
                bmi,
                (1 if ever_married == "Yes" else 0),
                [
                    "Government job",
                    "Never worked",
                    "Private",
                    "Self-employed",
                    "Children",
                ].index(work_type),
            ]
        ],
        columns=[
            "age",
            "hypertension",
            "avg_glucose_level",
            "bmi",
            "ever_married",
            "work_type",
        ],
    )

    # Handle unknown glucose level by setting it to NaN
    if unknown_glucose:
        input_data["avg_glucose_level"] = np.nan

    # Apply preprocessing steps to the input data
    input_data[["bmi", "avg_glucose_level"]] = imputer.transform(
        input_data[["bmi", "avg_glucose_level"]]
    )
    input_scaled = scaler.transform(input_data)

    # Make the prediction using the pre-trained model
    prediction = model.predict(input_scaled)

    # Display the prediction result
    st.markdown(
        '<div class="subheader">Prediction Result</div>', unsafe_allow_html=True
    )
    if prediction[0] == 1:
        st.markdown(
            '<div class="result error">High risk of stroke. Please consult a healthcare provider immediately.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result success">Low risk of stroke. Continue to maintain a healthy lifestyle.</div>',
            unsafe_allow_html=True,
        )

    # Add a disclaimer
    st.write(
        """
        Please note that this prediction is based on the provided information and should not be considered a definitive diagnosis.
        Always consult with a qualified healthcare professional for accurate assessment and advice.
        """
    )
