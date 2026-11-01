
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# Load trained model
model = joblib.load("my_model.pkl")


# Create prediction function
def predict(age, gender, attendance_rate, prior_gpa,
            lms_logins, study_hours,
            assignment_submission_rate, engagement_score):

    # Create input dataframe with correct feature order
    input_data = pd.DataFrame([[
        age,
        gender,
        attendance_rate,
        prior_gpa,
        lms_logins,
        study_hours,
        assignment_submission_rate,
        engagement_score
    ]], columns=[
        'Age',
        'Gender',
        'Attendance_Rate',
        'Prior_GPA',
        'LMS_Logins',
        'Study_Hours_Per_Week',
        'Assignment_Submission_Rate',
        'Engagement_Score'
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Calculate prediction confidence
    probability = model.predict_proba(input_data)[0]

    confidence = max(probability) * 100

    # Display predicted performance
    return (
        f"Predicted Student Performance: {prediction}\n"
        f"Prediction Confidence: {confidence:.2f}%"
    )


# Create Gradio interface
app = gr.Interface(

    fn=predict,

    inputs=[
        gr.Number(label="Age", value=20, precision=0),

        gr.Dropdown(
            choices=["Male", "Female"],
            label="Gender"
        ),

        gr.Number(label="Attendance Rate (%)", value=80),

        gr.Number(label="Prior GPA", value=3.0),

        gr.Number(label="LMS Logins", value=150, precision=0),

        gr.Number(label="Study Hours Per Week", value=15),

        gr.Number(
            label="Assignment Submission Rate (%)",
            value=80
        ),

        gr.Number(label="Engagement Score", value=50)
    ],

    outputs=gr.Textbox(
        label="Student Performance Prediction",
        lines=3
    ),

    title="Student Academic Performance Prediction System",

    description=(
        "Enter student details to predict "
        "their final academic performance "
        "as High, Medium or Low."
    )
)


# Launch application
if __name__ == "__main__":
    app.launch(share=True)
