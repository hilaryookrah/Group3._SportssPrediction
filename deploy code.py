import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib  # Replace pickle with joblib

# Create a Flask app
app = Flask(__name__, template_folder= "C:\\Users\\aryee\\OneDrive\\Midsem project(AI)\\templates")

# Load the model using joblib
model = joblib.load("C:\\Users\\aryee\\OneDrive\\Midsem project(AI)\\finalized_best_model (1).pkl")

# Define the route for the home page
@app.route("/")
def index():
    return render_template("page.html")

# Define the route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input features from the POST request
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    
    # Pass the result to an HTML template
    return render_template("page.html", prediction_text='The player rating is {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run()
