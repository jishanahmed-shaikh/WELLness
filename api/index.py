from flask import Flask, render_template, request, jsonify
import joblib
import re
import numpy as np
from datetime import datetime
import traceback
import os

app = Flask(__name__)

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load your ML model and TF-IDF vectorizer with error handling
try:
    model_path = os.path.join(parent_dir, 'model.pkl')
    vectorizer_path = os.path.join(parent_dir, 'tfidf_vectorizer.pkl')
    
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    print("Model and TF-IDF vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    tfidf_vectorizer = None

# Mental health status mapping for better display
STATUS_MAPPING = {
    "Anxiety": {"emoji": "😰", "color": "#FF6B6B", "description": "Feeling worried, nervous, or uneasy"},
    "Normal": {"emoji": "😊", "color": "#51CF66", "description": "Feeling balanced and well"},
    "Depression": {"emoji": "😔", "color": "#495057", "description": "Feeling sad, hopeless, or down"},
    "Suicidal": {"emoji": "🆘", "color": "#E03131", "description": "Having thoughts of self-harm - please seek help immediately"},
    "Bipolar": {"emoji": "🌓", "color": "#9775FA", "description": "Experiencing mood swings between high and low states"}
}

def preprocess_text(text):
    """Text preprocessing to match the training process"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (same as training)
    text = re.sub('[{}]'.format(re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')), '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_confidence_level(prediction_proba):
    """Calculate confidence level based on prediction probability"""
    if prediction_proba is None:
        return "Unknown"
    
    max_prob = np.max(prediction_proba)
    if max_prob >= 0.8:
        return "Very High"
    elif max_prob >= 0.6:
        return "High"
    elif max_prob >= 0.4:
        return "Medium"
    else:
        return "Low"

def predict_mental_health_status(text):
    """
    Predict mental health status using the trained model and TF-IDF vectorizer
    """
    if model is None or tfidf_vectorizer is None:
        raise ValueError("Model or vectorizer not loaded")
    
    # Preprocess the text (same as training)
    processed_text = preprocess_text(text)
    
    # Transform using the TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get prediction probabilities if available
    prediction_proba = None
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(text_vectorized)[0]
    
    return prediction, prediction_proba

@app.route('/')
def index():
    return render_template('index.html', 
                         prediction=None, 
                         statuses=STATUS_MAPPING,
                         emotions=STATUS_MAPPING,  # Keep for template compatibility
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/reset', methods=['GET'])
def reset():
    """Reset the form and clear all results"""
    return render_template('index.html', 
                         prediction=None, 
                         statuses=STATUS_MAPPING,
                         emotions=STATUS_MAPPING,  # Keep for template compatibility
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         reset_success=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or tfidf_vectorizer is None:
            return render_template('index.html', 
                                 error="Model or vectorizer not loaded. Please check if both model.pkl and tfidf_vectorizer.pkl exist.",
                                 statuses=STATUS_MAPPING,
                                 emotions=STATUS_MAPPING,  # Keep for template compatibility
                                 current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        user_input = request.form.get('text', '').strip()
        
        if not user_input:
            return render_template('index.html', 
                                 error="Please enter some text to analyze.",
                                 statuses=STATUS_MAPPING,
                                 emotions=STATUS_MAPPING,  # Keep for template compatibility
                                 current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        if len(user_input) < 3:
            return render_template('index.html', 
                                 error="Please enter at least 3 characters for better analysis.",
                                 user_input=user_input,
                                 statuses=STATUS_MAPPING,
                                 emotions=STATUS_MAPPING,  # Keep for template compatibility
                                 current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Make prediction using our custom function
        prediction, prediction_proba = predict_mental_health_status(user_input)
        
        # Get the processed text for display
        processed_text = preprocess_text(user_input)
        
        # Get confidence level
        confidence = get_confidence_level(prediction_proba)
        
        # Get status details
        status_info = STATUS_MAPPING.get(prediction, {
            "emoji": "🤔",
            "color": "#666666",
            "description": f"Status: {prediction}"
        })
        
        # Add the name to status_info
        status_info["name"] = prediction
        
        # Create probability breakdown for all classes if available
        prob_breakdown = None
        if prediction_proba is not None:
            # Get the class names from the model
            class_names = model.classes_ if hasattr(model, 'classes_') else list(STATUS_MAPPING.keys())
            prob_breakdown = []
            for i, class_name in enumerate(class_names):
                if i < len(prediction_proba):
                    prob_info = STATUS_MAPPING.get(class_name, {
                        "emoji": "🤔",
                        "color": "#666666",
                        "description": f"Status: {class_name}"
                    })
                    prob_breakdown.append({
                        "name": class_name,
                        "emoji": prob_info["emoji"],
                        "color": prob_info["color"],
                        "probability": prediction_proba[i]
                    })
        
        return render_template('index.html', 
                             prediction=prediction,
                             status_info=status_info,
                             emotion_info=status_info,  # Keep for template compatibility
                             confidence=confidence,
                             prediction_proba=prediction_proba.tolist() if prediction_proba is not None else None,
                             prob_breakdown=prob_breakdown,
                             user_input=user_input,
                             processed_text=processed_text,  # Add processed text for display
                             statuses=STATUS_MAPPING,
                             emotions=STATUS_MAPPING,  # Keep for template compatibility
                             current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    except Exception as e:
        error_msg = f"An error occurred during prediction: {str(e)}"
        print(f"Prediction error: {traceback.format_exc()}")
        
        return render_template('index.html', 
                             error=error_msg,
                             user_input=request.form.get('text', ''),
                             statuses=STATUS_MAPPING,
                             emotions=STATUS_MAPPING,  # Keep for template compatibility
                             current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if model is None or tfidf_vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded"}), 500
        
        # Make prediction using our custom function
        prediction, prediction_proba = predict_mental_health_status(text)
        
        # Get status details
        status_info = STATUS_MAPPING.get(prediction, {
            "emoji": "🤔",
            "color": "#666666",
            "description": f"Status: {prediction}"
        })
        
        return jsonify({
            "prediction": prediction,
            "status": status_info,
            "confidence": get_confidence_level(prediction_proba),
            "probabilities": prediction_proba.tolist() if prediction_proba is not None else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel compatibility
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
