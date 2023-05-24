from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
loaded_model = pickle.load(open("trained_model.pkl", "rb"))
loaded_model.feature_names = ['Age', 'Gender', 'Disease']  # Set the feature names

# Create the Flask application
app = Flask(__name__)


# Define the route and the corresponding function for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request body
    data = request.json

    # Create a DataFrame from the data
    df = pd.DataFrame([data])

    # Perform the necessary preprocessing
    for column in df.columns:
        if df[column].dtype == "object":
            label_encoder_x = LabelEncoder()  # Initialize LabelEncoder
            df[column] = label_encoder_x.fit_transform(df[column])

    # Make the prediction
    prediction = loaded_model.predict(df)

    # Convert the prediction to the desired JSON format
    response = {'prediction': int(prediction[0])}

    # Return the response as JSON
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
