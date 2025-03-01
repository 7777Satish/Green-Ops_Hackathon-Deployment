from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("energy_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"predicted_energy": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
