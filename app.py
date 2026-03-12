from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get user inputs
    amount = float(request.form["amount"])
    international = int(request.form["international"])
    chip = int(request.form["chip"])
    pin = int(request.form["pin"])
    hour = int(request.form["hour"])
    device = int(request.form["device"])

    features = np.array([[amount, international, chip, pin, hour, device]])

    # Model prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    risk_score = 0

    # -------------------------
    # Risk Rules
    # -------------------------

    if amount > 50000:
        risk_score += 2

    if international == 1:
        risk_score += 2

    if chip == 0:
        risk_score += 1

    if pin == 0:
        risk_score += 2

    if hour >= 0 and hour <= 5:
        risk_score += 1

    if device == 2:
        risk_score += 1

    # -------------------------
    # Final Decision
    # -------------------------

    if prediction[0] == 1 or probability > 0.6 or risk_score >= 4:
        result = "⚠ Fraud Transaction Detected"
    else:
        result = "✅ Legitimate Transaction"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)