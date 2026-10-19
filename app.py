import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling with elevated card structures and box shadows
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .header-card {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4), 0 8px 10px -6px rgba(79, 70, 229, 0.2);
        margin-bottom: 2rem;
    }
    .header-card h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-card p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    .result-card {
        background: #ffffff;
        border-left: 6px solid #10b981;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-card h2 {
        color: #111827;
        margin-bottom: 0.2rem;
    }
    .result-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10b981;
    }
    div[data-testid="stForm"] button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }
    div[data-testid="stForm"] button:hover {
        background-color: #4338ca !important;
        box-shadow: 0 6px 12px -2px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-1px);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Render main page title header
st.markdown(
    """
    <div class="header-card">
        <h1>🎓 Student Performance Prediction</h1>
        <p>Enter the student parameters below to predict the estimated exam performance index.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Function to load model with caching
@st.cache_resource
def load_model(file_path="model.pkl"):
    if not os.path.exists(file_path):
        st.error(
            f"Model file `{file_path}` not found in the root directory. Please verify your file structure."
        )
        return None
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model("model.pkl")

# Main Form Interface
with st.form("prediction_form"):
    st.subheader("📋 Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age", min_value=10, max_value=100, value=20, step=1
        )
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        course = st.selectbox(
            "Course", options=["Engineering", "Science", "Arts", "Commerce"]
        )
        study_hours = st.number_input(
            "Study Hours per Week",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
        )

    with col2:
        class_attendance = st.slider(
            "Class Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
        )
        internet_access = st.selectbox(
            "Internet Access", options=["Yes", "No"]
        )
        sleep_hours = st.number_input(
            "Sleep Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )
        sleep_quality = st.select_slider(
            "Sleep Quality",
            options=["Poor", "Average", "Good", "Excellent"],
            value="Good",
        )

    with col3:
        study_method = st.selectbox(
            "Study Method",
            options=["Self Study", "Group Study", "Online Lectures", "Tutor"],
        )
        facility_rating = st.slider(
            "Facility Rating (1-5)", min_value=1, max_value=5, value=3, step=1
        )
        exam_difficulty = st.select_slider(
            "Exam Difficulty",
            options=["Easy", "Moderate", "Hard"],
            value="Moderate",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🚀 Generate Prediction")

# Process inputs and output prediction on submit
if submit_button:
    if model is None:
        st.warning("Prediction skipped: Model file could not be loaded.")
    else:
        # Categorical variable mapping (Adjust keys/values to match training script schema)
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        course_map = {"Engineering": 0, "Science": 1, "Arts": 2, "Commerce": 3}
        internet_map = {"No": 0, "Yes": 1}
        sleep_quality_map = {
            "Poor": 0,
            "Average": 1,
            "Good": 2,
            "Excellent": 3,
        }
        study_method_map = {
            "Self Study": 0,
            "Group Study": 1,
            "Online Lectures": 2,
            "Tutor": 3,
        }
        exam_difficulty_map = {"Easy": 0, "Moderate": 1, "Hard": 2}

        # Build feature vector input array
        features = np.array(
            [
                [
                    age,
                    gender_map[gender],
                    course_map[course],
                    study_hours,
                    class_attendance,
                    internet_map[internet_access],
                    sleep_hours,
                    sleep_quality_map[sleep_quality],
                    study_method_map[study_method],
                    facility_rating,
                    exam_difficulty_map[exam_difficulty],
                ]
            ]
        )

        try:
            prediction = model.predict(features)[0]

            # Display formatted output
            st.markdown(
                f"""
                <div class="result-card">
                    <h2>Predicted Exam Score</h2>
                    <div class="result-value">{prediction:.2f}</div>
                    <p style="color: #6b7280; margin-top: 0.5rem;">Value calculated using Support Vector Regression model.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        except Exception as err:
            st.error(f"Error executing SVR model prediction: {err}")
