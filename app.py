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

# Input fields for each parameter with corrected types and ranges
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0, step=1)  
Glucose = st.number_input("Glucose", min_value=0, max_value=199, value=120, step=1)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=122, value=70, step=1)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=99, value=20, step=1)
Insulin = st.number_input("Insulin", min_value=0, max_value=846, value=79, step=1)
BMI = st.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.078, max_value=2.42, value=0.471, step=0.001)
Age = st.number_input("Age", min_value=21, max_value=81, value=33, step=1)



# Map the gender and embarked values to numeric
gender_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'BloodPressure' : BloodPressure,
    'SkinThickness': SkinThickness, # Corrected key
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_Outcome(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: Dibatic with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: Did not Dibatic with probability {prob:.2f}")
