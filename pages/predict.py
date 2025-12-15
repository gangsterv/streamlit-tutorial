import streamlit as st
import joblib
import pandas as pd

# Load the model
pipeline = joblib.load("models/model_joblib.pkl")

st.title("Titantic Prediction")
st.write("This is the prediction page.")

# Ask the user for input data
st.write("\nPlease enter your information below.")

age = st.select_slider("Age", options=list(range(0, 120)))
st.select_slider("SibSp", options=list(range(0, 10)), key="sibsp")
st.select_slider("Parch", options=list(range(0, 10)), key="parch")

st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.01, key="fare")

st.checkbox("Boarded on Queenstwon?", key="q")
st.checkbox("Boarded on Southampton?", key="s")

st.radio("Sex", options=["Male", "Female"], key='sex')

if st.button("Predict", key="predict_button"):
    # Collect the data from the widgets
    data = [
        age,
        st.session_state.sibsp,
        st.session_state.parch,
        st.session_state.fare,
        1 if st.session_state.sex == 'Male' else 0,
        st.session_state.q,
        st.session_state.s,
    ]


    # Create a pandas dataframe out of the list
    data = pd.DataFrame([data], columns=['Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S'])

    # Make a prediction
    prediction = pipeline.predict(data)

    st.write(f"### The prediction of your survival is: {bool(prediction)}")