import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request
import pandas as pd
from PIL import Image
import joblib
import numpy as np
from model_utils import extract_features_from_image

app = Flask(__name__)

# LOAD TRAINED ARTIFACTS

pipeline = joblib.load("model.pkl")
selected_features = joblib.load("selected_features.pkl")
feature_means = joblib.load("feature_means.pkl")
asd_profile = joblib.load("asd_profile.pkl")
normal_profile = joblib.load("normal_profile.pkl")


# ROUTES

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None or file.filename == "":
        return render_template(
            "index.html",
            error="No image file uploaded. Please select an image."
        )

    try:
        image = Image.open(file)

        # 1. EXTRACT FEATURES
        data = extract_features_from_image(image)
        df = pd.DataFrame([data])

        # 2. ALIGN FEATURES
        df = df.reindex(columns=selected_features)

        for col in selected_features:
            if pd.isna(df[col].iloc[0]):
                if col in feature_means.index:
                    df[col] = feature_means[col]
                else:
                    df[col] = 0

        # 3. PREDICTION
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]

        # 4. CONFIDENCE INTERPRETATION
        if probability >= 0.7:
            confidence_text = "High likelihood of ASD traits"
            confidence_level = "high"
        elif probability >= 0.5:
            confidence_text = "Moderate likelihood of ASD traits"
            confidence_level = "moderate"
        elif probability >= 0.3:
            confidence_text = "Low likelihood of ASD traits"
            confidence_level = "low"
        else:
            confidence_text = "Unlikely to show ASD traits"
            confidence_level = "very-low"

        # 5 RETURN RESULT
        return render_template(
            "index.html",
            prediction=int(prediction),
            probability=round(probability, 3),
            confidence_text=confidence_text,
            confidence_level=confidence_level,
            features=data,
            selected_features=selected_features,
            filename=file.filename,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error processing image: {str(e)}"
        )

# RUN APP

if __name__ == "__main__":
    app.run(debug=True)