import streamlit as st
import pandas as pd

st.title("Titanic Survival Prediction")
st.write("This app predicts whether a passenger survived the Titanic disaster based on their features.\n\nHere is a sample of the data:")

df = pd.read_csv("data/titanic_preprocessed.csv")
st.write(df.head())