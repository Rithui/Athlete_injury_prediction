"""
Athlete Injury Recovery Prediction System
Flask Web Application - Optimized Version
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model components
MODEL_PATH = 'models/recovery_prediction_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
SELECTOR_PATH = 'models/selector.pkl'
FEATURE_INFO_PATH = 'models/feature_info.pkl'

model = None
scaler = None
selector = None
feature_info = None

def load_model():
    """Load the trained model and preprocessing components"""
    global model, scaler, selector, feature_info
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH)
        print("✅ Model loaded successfully!")
        print(f"   Model: {feature_info.get('model_name', 'Unknown')}")
        print(f"   Accuracy: {feature_info.get('accuracy', 0)*100:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def engineer_features(data):
    """Apply feature engineering to input data"""
    X = pd.DataFrame([data])
    
    # Create engineered features (same as training)
    X['Training_Load'] = X['Training_Frequency'] * X['Training_Duration'] * X['Training_Intensity'] / 100
    X['Recovery_Ratio'] = X['Recovery_Time'] / (X['Training_Duration'] + 1)
    X['Sleep_Recovery_Score'] = X['Sleep_Hours'] * X['Recovery_Time'] / 100
    X['Risk_Index'] = (X['Injury_History'] * X['Training_Intensity']) / (X['Flexibility_Score'] + 1)
    X['Physical_Stress'] = X['Stress_Level'] * X['Muscle_Asymmetry'] / 10
    X['Warmup_Ratio'] = X['Warmup_Time'] / (X['Training_Duration'] + 1)
    X['BMI_Age_Factor'] = X['BMI'] * X['Age'] / 100
    X['Fitness_Score'] = (X['Flexibility_Score'] * X['Sleep_Hours']) / (X['Stress_Level'] + 1)
    X['Training_Efficiency'] = X['Training_Intensity'] / (X['Training_Frequency'] + 1)
    X['Recovery_Need'] = X['Training_Load'] / (X['Sleep_Hours'] + 1)
    
    # Polynomial features
    X['Injury_History_Sq'] = X['Injury_History'] ** 2
    X['Training_Intensity_Sq'] = X['Training_Intensity'] ** 2
    X['Sleep_Hours_Sq'] = X['Sleep_Hours'] ** 2
    
    # Categorical binning
    X['Age_Group'] = pd.cut(X['Age'], bins=[0, 25, 30, 35, 100], labels=[0, 1, 2, 3]).astype(int)
    X['BMI_Category'] = pd.cut(X['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    
    return X

@app.route('/')
def home():
    """Render the home page"""
    accuracy = feature_info['accuracy'] * 100 if feature_info else 0
    model_name = feature_info.get('model_name', 'SVM') if feature_info else 'SVM'
    return render_template('index.html', accuracy=accuracy, model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        
        # Extract features
        input_data = {
            'Age': float(data['age']),
            'Gender': float(data['gender']),
            'Height_cm': float(data['height']),
            'Weight_kg': float(data['weight']),
            'BMI': float(data['bmi']),
            'Training_Frequency': float(data['training_frequency']),
            'Training_Duration': float(data['training_duration']),
            'Warmup_Time': float(data['warmup_time']),
            'Sleep_Hours': float(data['sleep_hours']),
            'Flexibility_Score': float(data['flexibility_score']),
            'Muscle_Asymmetry': float(data['muscle_asymmetry']),
            'Recovery_Time': float(data['recovery_time']),
            'Injury_History': float(data['injury_history']),
            'Stress_Level': float(data['stress_level']),
            'Training_Intensity': float(data['training_intensity'])
        }
        
        # Apply feature engineering
        X_engineered = engineer_features(input_data)
        
        # Scale features
        X_scaled = scaler.transform(X_engineered)
        
        # Select features
        X_selected = selector.transform(X_scaled)
        
        # Make prediction
        prediction = model.predict(X_selected)[0]
        probability = model.predict_proba(X_selected)[0]
        
        # Calculate recovery insights
        extended_recovery_prob = probability[1] * 100
        quick_recovery_prob = probability[0] * 100
        
        # Determine recovery outlook and timeline
        if extended_recovery_prob < 25:
            recovery_outlook = "Excellent"
            outlook_class = "excellent"
            estimated_days = "3-5 days"
            recommendations = [
                "Your recovery profile is excellent - continue current practices",
                "Maintain your current sleep schedule of {:.1f} hours".format(input_data['Sleep_Hours']),
                "Your warm-up routine is effective - keep it up",
                "Consider light active recovery exercises",
                "Stay hydrated and maintain balanced nutrition"
            ]
        elif extended_recovery_prob < 50:
            recovery_outlook = "Good"
            outlook_class = "good"
            estimated_days = "5-10 days"
            recommendations = [
                "Good recovery potential with minor optimizations needed",
                "Consider increasing sleep to 8+ hours per night",
                "Add 5-10 minutes to your warm-up routine",
                "Include stretching and mobility work post-training",
                "Monitor any muscle soreness or fatigue closely",
                "Consider reducing training intensity by 10-15%"
            ]
        elif extended_recovery_prob < 75:
            recovery_outlook = "Moderate"
            outlook_class = "moderate"
            estimated_days = "10-21 days"
            recommendations = [
                "Recovery may take longer - proactive measures recommended",
                "Prioritize sleep quality - aim for 8-9 hours nightly",
                "Reduce training frequency by 1-2 days per week",
                "Extend warm-up time to at least 15-20 minutes",
                "Focus on flexibility and mobility exercises daily",
                "Consider professional physiotherapy assessment",
                "Address muscle asymmetry with targeted exercises",
                "Implement stress management techniques"
            ]
        else:
            recovery_outlook = "Extended"
            outlook_class = "extended"
            estimated_days = "21-42 days"
            recommendations = [
                "⚠️ High risk of extended recovery - immediate action required",
                "Strongly recommend reducing training load by 50%",
                "Consult a sports medicine professional immediately",
                "Prioritize complete rest and recovery protocols",
                "Address underlying factors: stress, sleep, flexibility",
                "Consider temporary training modification or break",
                "Implement comprehensive rehabilitation program",
                "Focus on nutrition and anti-inflammatory foods",
                "Daily stretching and foam rolling essential",
                "Monitor symptoms closely - seek help if worsening"
            ]
        
        # Generate detailed analysis
        analysis = {
            'training_load_status': 'High' if input_data['Training_Frequency'] * input_data['Training_Duration'] > 400 else 'Moderate' if input_data['Training_Frequency'] * input_data['Training_Duration'] > 200 else 'Low',
            'sleep_status': 'Optimal' if input_data['Sleep_Hours'] >= 7.5 else 'Suboptimal' if input_data['Sleep_Hours'] >= 6 else 'Insufficient',
            'flexibility_status': 'Good' if input_data['Flexibility_Score'] >= 7 else 'Average' if input_data['Flexibility_Score'] >= 5 else 'Needs Improvement',
            'stress_status': 'Low' if input_data['Stress_Level'] <= 3 else 'Moderate' if input_data['Stress_Level'] <= 6 else 'High',
            'injury_history_impact': 'Low' if input_data['Injury_History'] <= 1 else 'Moderate' if input_data['Injury_History'] <= 3 else 'High'
        }
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'extended_recovery_probability': round(extended_recovery_prob, 2),
            'quick_recovery_probability': round(quick_recovery_prob, 2),
            'recovery_outlook': recovery_outlook,
            'outlook_class': outlook_class,
            'estimated_recovery_days': estimated_days,
            'recommendations': recommendations,
            'analysis': analysis,
            'model_confidence': round(max(probability) * 100, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/about')
def about():
    """Render the about page"""
    accuracy = feature_info['accuracy'] * 100 if feature_info else 0
    model_name = feature_info.get('model_name', 'Ensemble') if feature_info else 'Ensemble'
    return render_template('about.html', accuracy=accuracy, model_name=model_name)

@app.route('/api/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'accuracy': feature_info['accuracy'] if feature_info else None,
        'model_name': feature_info.get('model_name') if feature_info else None
    })

if __name__ == '__main__':
    if load_model():
        print("\n🚀 Starting Athlete Injury Recovery Prediction System...")
        print("📍 Access the application at: http://127.0.0.1:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run train_model.py first.")
