from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return "CKD Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input to array
    features = np.array(data['features']).reshape(1, -1)

    # Scale input
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)