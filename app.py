# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import matplotlib
import numpy as np
from privacy_preserving import preprocess_data_with_privacy

matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = [data['url']]
    
    # Transform the URL using the vectorizer
    features = vectorizer.transform(url)
    
    # Apply privacy-preserving preprocessing
    features_privacy = preprocess_data_with_privacy(features.toarray(), epsilon=0.1)
    
    prediction = model.predict(features_privacy)

    # Plot the privacy-preserved features
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(features_privacy[0])), features_privacy[0])
    plt.xlabel('Feature Index')
    plt.ylabel('Privacy-Preserved Feature Value')
    plt.title('Privacy-Preserved Features')
    plot_path = 'static/privacy_preserved_features.png'
    plt.savefig(plot_path)
    plt.close()
    
    return jsonify({
        'prediction': int(prediction[0]),
        'features_privacy': features_privacy.tolist(),
        'plot_path': plot_path
    })

if __name__ == "__main__":
    app.run(debug=True)
