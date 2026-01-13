import streamlit as st
import pickle
import numpy as np

# Load the saved model
try:
    model = pickle.load(open('heart_disease_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please run 'train_model.py' first.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    </style>
""", unsafe_allow_html=True)

st.title('❤️ Heart Disease Prediction App')
st.write("Enter the following details to predict the likelihood of heart disease.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', options=[
        'Typical Angina (0)', 
        'Atypical Angina (1)', 
        'Non-anginal Pain (2)', 
        'Asymptomatic (3)'
    ])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=300, value=120)
    chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['False', 'True'])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[
        'Normal (0)', 
        'ST-T Wave Abnormality (1)', 
        'Left Ventricular Hypertrophy (2)'
    ])

with col2:
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
    exang = st.selectbox('Exercise Induced Angina', options=['No', 'Yes'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[
        'Upsloping (0)', 
        'Flat (1)', 
        'Downsloping (2)'
    ])
    ca = st.selectbox('Number of Major Vessels (0-4)', options=[0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia', options=[
        'Normal (0)', 
        'Fixed Defect (1)', 
        'Reversable Defect (2)', 
        'Defect (3)'
    ])

# Preprocessing for prediction
def preprocess_input(sex, cp, fbs, restecg, exang, slope, thal):
    # Mapping categorical values to numeric
    sex_val = 1 if sex == 'Male' else 0
    cp_val = int(cp.split('(')[1].split(')')[0])
    fbs_val = 1 if fbs == 'True' else 0
    restecg_val = int(restecg.split('(')[1].split(')')[0])
    exang_val = 1 if exang == 'Yes' else 0
    slope_val = int(slope.split('(')[1].split(')')[0])
    thal_val = int(thal.split('(')[1].split(')')[0])
    
    return [sex_val, cp_val, fbs_val, restecg_val, exang_val, slope_val, thal_val]

# Prediction logic
if st.button('Predict Heart Disease'):
    sex_v, cp_v, fbs_v, restecg_v, exang_v, slope_v, thal_v = preprocess_input(sex, cp, fbs, restecg, exang, slope, thal)
    
    input_data = (age, sex_v, cp_v, trestbps, chol, fbs_v, restecg_v, thalach, exang_v, oldpeak, slope_v, ca, thal_v)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = model.predict(input_data_reshaped)
    prob = model.predict_proba(input_data_reshaped)[0][1]
    
    st.subheader('Prediction Result:')
    if prediction[0] == 0:
        st.success(f'The person presumably does NOT have heart disease. (Probability: {1-prob:.2f})')
    else:
        st.error(f'The person presumably HAS heart disease. (Probability: {prob:.2f})')

st.markdown("---")
st.write("Disclaimer: This is a machine learning model for educational purposes only and should not be used for medical diagnosis.")
