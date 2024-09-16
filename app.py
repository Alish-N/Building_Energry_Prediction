from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://66e81ded0c425124fd00aed9--darling-peony-261113.netlify.app"}})

# Load the saved model, scaler, and encoder
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

# Define numerical and categorical features
numerical_features = ['Building_Age', 'Number_of_Floors', 'Total_Area', 'Number_of_Windows', 
                      'Average_Window_Size', 'Total_Insulation_Area', 'Heating_Cooling_System_Efficiency']
categorical_features = ['Insulation_Type', 'Heating_System', 'Cooling_System']

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json(force=True)

    # Convert to DataFrame
    input_data = pd.DataFrame([data])

    # Compute new features if not provided
    if 'Average_Window_Size' not in input_data.columns:
        input_data['Average_Window_Size'] = input_data['Total_Area'] / input_data['Number_of_Windows']
    if 'Total_Insulation_Area' not in input_data.columns:
        input_data['Total_Insulation_Area'] = input_data['Total_Area'] * 0.5
    if 'Heating_Cooling_System_Efficiency' not in input_data.columns:
        input_data['Heating_Cooling_System_Efficiency'] = np.random.uniform(low=0.8, high=1.0, size=1)

    # Scale numerical features
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Encode categorical features
    one_hot_encoded = encoder.transform(input_data[categorical_features])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Concatenate scaled numerical features and encoded categorical features
    input_encoded = pd.concat([input_data[numerical_features].reset_index(drop=True), one_hot_df], axis=1)

    # Handle missing columns that may arise due to 'handle_unknown' set to 'ignore'
    missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0  # Add missing columns with zeros

    # Ensure the order of columns matches the training data
    input_encoded = input_encoded[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_encoded)

    # Return the prediction as JSON
    return jsonify({'Energy_Consumption': prediction[0]})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
