from flask import Flask, render_template, request
import pandas as pd
from PIL import Image
import joblib

from model_utils import extract_features_from_image

app = Flask(__name__)

# Load Trained Model

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    image = Image.open(file)

    save_path = "training_images/" + file.filename
    file.save(save_path)

    # Extract features using OCR
    data = extract_features_from_image(image)

    df = pd.DataFrame([data])

    # Handle missing values
    df = df.fillna(df.mean())

    # Scale using trained scaler
    X_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(probability,3),
        features=data,
        filename=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)