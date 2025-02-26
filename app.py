from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        gender = request.form['gender']
        hemoglobin = float(request.form['hemoglobin'])
        mcv = float(request.form['mcv'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])

        # Convert gender to numerical format
        gender_encoded = 1 if gender == "Male" else 0

        # Prepare input data
        input_data = np.array([[gender_encoded, hemoglobin, mcv, mch, mchc]])
        prediction = model.predict(input_data)[0]

        # Set prediction message
        prediction_text = "The patient is likely to have Anemia." if prediction == 1 else "The patient is unlikely to have Anemia."

        return render_template('predict.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error: {e}")

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
