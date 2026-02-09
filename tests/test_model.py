import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the model
pipeline = joblib.load("models/model_joblib.pkl")

# Create a pandas dataframe out of the list
data = pd.read_csv("data/test_data.csv")

X = data.drop(columns=["Survived"])
y = data["Survived"]

# Make a prediction
prediction = pipeline.predict(X)

# score = pipeline.score(X)
score = accuracy_score(y, prediction)

print("Score is:", score)

assert score > 0.95

