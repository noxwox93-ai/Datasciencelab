'''
Aim
To develop a Machine Learning–based prediction system using Logistic Regression and deploy it as a Flask API for predicting outcomes based on input features.
# Description

This program develops a Machine Learning prediction system using Logistic Regression and Flask API. 
The dataset is loaded and used to train the model by separating input features and target values.
 After training, the model is saved using Joblib for future predictions. 
 Flask is used to create API routes for checking server status and predicting outputs based on user input. 
 The system returns prediction results in JSON format through the API.

'''
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # input list

    features = np.array(data).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)



import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully!")

