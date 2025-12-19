import numpy as np
import pandas as pd

def predict_placement(model, scaler, input_data, feature_names):
    """
    Predict placement status and probability for a new student
    """

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode input
    input_df = pd.get_dummies(input_df)

    # Ensure same feature order as training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability
