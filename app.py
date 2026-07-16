from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("customer_churn_model.pkl")



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form["gender"]
    SeniorCitizen = int(request.form["SeniorCitizen"])
    Partner = request.form["Partner"]
    Dependents = request.form["Dependents"]
    tenure = int(request.form["tenure"])
    PhoneService = request.form["PhoneService"]
    MultipleLines = request.form["MultipleLines"]
    InternetService = request.form["InternetService"]
    OnlineSecurity = request.form["OnlineSecurity"]
    OnlineBackup = request.form["OnlineBackup"]
    DeviceProtection = request.form["DeviceProtection"]
    TechSupport = request.form["TechSupport"]
    StreamingTV = request.form["StreamingTV"]
    StreamingMovies = request.form["StreamingMovies"]
    Contract = request.form["Contract"]
    PaperlessBilling = request.form["PaperlessBilling"]
    PaymentMethod = request.form["PaymentMethod"]
    MonthlyCharges = float(request.form["MonthlyCharges"])
    TotalCharges = float(request.form["TotalCharges"])

    input_data = pd.DataFrame({

        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]

    })

    prediction = model.predict(input_data)[0]
    
   
    probability = model.predict_proba(input_data)[0][1]

    probability = model.predict_proba(input_data)[0][1]
    probability_percent = round(probability * 100, 2)

    if probability_percent >= 70:
        risk = "🔴 High Risk"
    elif probability_percent >= 40:
        risk = "🟡 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    if prediction == "Yes":
        result = "Customer is likely to Churn"
        color = "danger"
    else:
        result = "Customer is likely to Stay"
        color = "success"

    return render_template(
        "index.html",
        prediction=result,
        probability=f"{probability_percent}%",
        probability_value=probability_percent,
        color=color,
        risk=risk
)

    return render_template(
        "index.html",
        prediction=result,
        probability=f"{probability:.2%}",
        color=color
    )


if __name__ == "__main__":
    app.run(debug=True)