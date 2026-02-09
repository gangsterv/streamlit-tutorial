import joblib
import pandas as pd

# Load the model
pipeline = joblib.load("models/model_joblib.pkl")

# Create a pandas dataframe out of the list
data = pd.read_csv("../data/test_data.csv")

# Make a prediction
# prediction = pipeline.predict(data)

score = pipeline.score(data)

print("Score is:", score)

assert score > 0.6

