import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_Outcome(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("Dibatic Prediction")

# Input fields for each parameter
Pregnancies = st.selectbox("Pregnancies", min_value=0.0, max_value=100.0,value=1.0)
BloodPressure = st.selectbox("BloodPressure",min_value=.50.0,max_value=198.0,value=1.0)
skinThSickness = st.number_input("skinThSickness", min_value=0.0,max_value=100.0,value=50.0, step=0.1)
Insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=1.0)
BMI= st.number_input("BMI", min_value=0.0, max_value=100.0, value=1.0)
DiabetesPedigreeFunction= st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=500.0, value=7.25, step=0.1)
Age = st.selectbox("Age",min_value=0.0,max_value=100.0,value=50.0, step=0.1)

# Map the gender and embarked values to numeric
gender_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'BloodPressure' : BloodPressure,
    'skinThSickness': skinThSickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_survival(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: Dibatic with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: Did not Dibatic with probability {prob:.2f}")
