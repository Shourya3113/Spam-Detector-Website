# app.py
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Configure CORS ---
# This allows the index.html file on your computer to talk to this server.
CORS(app)

# --- Build reliable paths to model files ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
model_path = os.path.join(script_dir, 'model.pkl')


# --- Load The Machine Learning Models ---
try:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(">>> Vectorizer and model loaded successfully.")
except FileNotFoundError:
    print("="*50)
    print(f"ERROR: Could not find '{vectorizer_path}' or '{model_path}'")
    print("Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory as app.py")
    print("="*50)
    vectorizer = None
    model = None

# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or vectorizer not loaded on the server.'}), 500

    try:
        data = request.get_json()
        message = data['message']
        message_vector = vectorizer.transform([message])
        prediction_code = model.predict(message_vector)[0]
        result = 'spam' if prediction_code == 1 else 'not_spam'
        print(f">>> Prediction for message: '{message[:30]}...' -> {result.upper()}")
        return jsonify({'prediction': result})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Define a Root Endpoint ---
@app.route('/')
def home():
    return "<h1>Local Spam Detection Server</h1><p>This server is running correctly.</p>"

# --- Run the App ---
if __name__ == '__main__':
    print(">>> Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=False)

