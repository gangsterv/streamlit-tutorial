import joblib
import pandas as pd

# Load the model
pipeline = joblib.load("../models/model_joblib.pkl")

print(pipeline)

# Example input data for prediction
data = [38.0, 1, 0, 71.2833, 0, 0, 0]

# Create a pandas dataframe out of the list
data = pd.DataFrame([data], columns=['Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S'])

# Make a prediction
prediction = pipeline.predict(data)
print(f"Predicted class: {prediction[0]}")

