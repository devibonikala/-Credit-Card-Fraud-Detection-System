from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Define the expected features in exact order (matches your dataset)
EXPECTED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

@app.route('/')
def home():
    return render_template('index.html', features=EXPECTED_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate all required fields are present
        missing_fields = [field for field in EXPECTED_FEATURES if field not in request.form]
        if missing_fields:
            raise ValueError(f"Missing fields: {', '.join(missing_fields)}")

        # Get all features in correct order and convert to float
        features = []
        for field in EXPECTED_FEATURES:
            value = request.form[field]
            if not value.strip():
                raise ValueError(f"Empty value for {field}")
            try:
                features.append(float(value))
            except ValueError:
                raise ValueError(f"Invalid number for {field}: {value}")

        # Verify we have exactly 30 features (Time + V1-V28 + Amount)
        if len(features) != 30:
            raise ValueError(f"Expected 30 features but got {len(features)}")

        # Scale features and make prediction
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)

        # Return result
        result = "❌ Fraudulent Transaction Detected!" if prediction[0] == 1 else "✅ Transaction is Safe."
        return render_template('index.html', prediction_text=result, features=EXPECTED_FEATURES)

    except ValueError as e:
        error_msg = f"⚠️ Input Error: {str(e)}"
        return render_template('index.html', prediction_text=error_msg, features=EXPECTED_FEATURES)
    except Exception as e:
        error_msg = f"⚠️ System Error: {str(e)}"
        return render_template('index.html', prediction_text=error_msg, features=EXPECTED_FEATURES)

if __name__ == '__main__':
    app.run(debug=True)