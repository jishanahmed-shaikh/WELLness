from flask import Flask, render_template, request, jsonify
import joblib
import re
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)

# Load your ML model with error handling
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Emotion mapping for better display
EMOTION_MAPPING = {
    0: {"name": "Sadness", "emoji": "😢", "color": "#4A90E2", "description": "Feeling down or melancholic"},
    1: {"name": "Joy", "emoji": "😊", "color": "#7ED321", "description": "Feeling happy and positive"},
    2: {"name": "Love", "emoji": "❤️", "color": "#D0021B", "description": "Feeling affectionate and caring"},
    3: {"name": "Anger", "emoji": "😠", "color": "#F5A623", "description": "Feeling frustrated or irritated"},
    4: {"name": "Fear", "emoji": "😨", "color": "#9013FE", "description": "Feeling anxious or worried"},
    5: {"name": "Surprise", "emoji": "😲", "color": "#50E3C2", "description": "Feeling amazed or shocked"}
}

def preprocess_text(text):
    """Basic text preprocessing"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
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

@app.route('/')
def index():
    return render_template('index.html', 
                         prediction=None, 
                         emotions=EMOTION_MAPPING,
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/reset', methods=['GET'])
def reset():
    """Reset the form and clear all results"""
    return render_template('index.html', 
                         prediction=None, 
                         emotions=EMOTION_MAPPING,
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         reset_success=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please check if model.pkl exists.",
                                 emotions=EMOTION_MAPPING,
                                 current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        user_input = request.form.get('text', '').strip()
        
        if not user_input:
            return render_template('index.html', 
                                 error="Please enter some text to analyze.",
                                 emotions=EMOTION_MAPPING,
                                 current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        if len(user_input) < 3:
            return render_template('index.html', 
                                 error="Please enter at least 3 characters for better analysis.",
                                 user_input=user_input,
                                 emotions=EMOTION_MAPPING,
                                 current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Preprocess the text
        processed_text = preprocess_text(user_input)
        
        # Try multiple input formats to handle different model types
        prediction = None
        input_format_used = None
        
        # Format 1: Simple list (most common for text models)
        try:
            input_data = [processed_text]
            prediction = model.predict(input_data)[0]
            input_format_used = "list"
        except Exception as e1:
            print(f"Format 1 failed: {e1}")
            
            # Format 2: 2D numpy array (for sklearn vectorized models)
            try:
                input_data = np.array([processed_text]).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                input_format_used = "2d_array"
            except Exception as e2:
                print(f"Format 2 failed: {e2}")
                
                # Format 3: Try with the raw text directly
                try:
                    prediction = model.predict([user_input])[0]
                    input_format_used = "raw_text"
                except Exception as e3:
                    print(f"Format 3 failed: {e3}")
                    
                    # Format 4: Try with different array shapes
                    try:
                        input_data = np.array([[processed_text]])
                        prediction = model.predict(input_data)[0]
                        input_format_used = "nested_array"
                    except Exception as e4:
                        print(f"Format 4 failed: {e4}")
                        
                        # Format 5: Try with single string
                        try:
                            prediction = model.predict(processed_text)[0]
                            input_format_used = "single_string"
                        except Exception as e5:
                            print(f"Format 5 failed: {e5}")
                            
                            # If all formats fail, provide detailed error
                            error_details = f"""
                            Model input format not compatible. Tried formats:
                            1. List format: {str(e1)[:100]}...
                            2. 2D array format: {str(e2)[:100]}...
                            3. Raw text format: {str(e3)[:100]}...
                            4. Nested array format: {str(e4)[:100]}...
                            5. Single string format: {str(e5)[:100]}...
                            
                            Please check how your model was trained and what input format it expects.
                            """
                            raise ValueError(error_details)
        
        if prediction is None:
            raise ValueError("Could not get prediction from model")
        
        print(f"Successful prediction using format: {input_format_used}")
        
        # Get prediction probabilities using the same successful format
        prediction_proba = None
        confidence = "Medium"
        try:
            if hasattr(model, 'predict_proba'):
                if input_format_used == "list":
                    prediction_proba = model.predict_proba([processed_text])[0]
                elif input_format_used == "2d_array":
                    input_data = np.array([processed_text]).reshape(1, -1)
                    prediction_proba = model.predict_proba(input_data)[0]
                elif input_format_used == "raw_text":
                    prediction_proba = model.predict_proba([user_input])[0]
                elif input_format_used == "nested_array":
                    input_data = np.array([[processed_text]])
                    prediction_proba = model.predict_proba(input_data)[0]
                elif input_format_used == "single_string":
                    prediction_proba = model.predict_proba(processed_text)[0]
                
                confidence = get_confidence_level(prediction_proba)
        except Exception as prob_error:
            print(f"Could not get prediction probabilities: {prob_error}")
            pass
        
        # Get emotion details
        emotion_info = EMOTION_MAPPING.get(prediction, {
            "name": f"Emotion {prediction}",
            "emoji": "🤔",
            "color": "#666666",
            "description": "Unknown emotion detected"
        })
        
        return render_template('index.html', 
                             prediction=prediction,
                             emotion_info=emotion_info,
                             confidence=confidence,
                             prediction_proba=prediction_proba,
                             user_input=user_input,
                             processed_text=processed_text,
                             emotions=EMOTION_MAPPING,
                             current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    except Exception as e:
        error_msg = f"An error occurred during prediction: {str(e)}"
        print(f"Prediction error: {traceback.format_exc()}")
        
        return render_template('index.html', 
                             error=error_msg,
                             user_input=request.form.get('text', ''),
                             emotions=EMOTION_MAPPING,
                             current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        processed_text = preprocess_text(text)
        
        # Use the same prediction logic as the main predict function
        try:
            input_data = [processed_text]
            prediction = model.predict(input_data)[0]
            
        except ValueError as ve:
            if "Expected 2D array" in str(ve):
                try:
                    input_array = np.array([processed_text]).reshape(1, -1)
                    prediction = model.predict(input_array)[0]
                except:
                    try:
                        input_data = np.array([[processed_text]])
                        prediction = model.predict(input_data)[0]
                    except:
                        raise ValueError("Model input format not compatible.")
            else:
                raise ve
        
        try:
            if hasattr(model, 'predict_proba'):
                if 'input_array' in locals():
                    prediction_proba = model.predict_proba(input_array)[0].tolist()
                elif 'input_data' in locals():
                    prediction_proba = model.predict_proba(input_data)[0].tolist()
                else:
                    prediction_proba = model.predict_proba([processed_text])[0].tolist()
            else:
                prediction_proba = None
        except:
            prediction_proba = None
        
        emotion_info = EMOTION_MAPPING.get(prediction, {
            "name": f"Emotion {prediction}",
            "emoji": "🤔",
            "color": "#666666",
            "description": "Unknown emotion detected"
        })
        
        return jsonify({
            "prediction": int(prediction),
            "emotion": emotion_info,
            "confidence": get_confidence_level(prediction_proba),
            "probabilities": prediction_proba
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)