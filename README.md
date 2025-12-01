# 🏥 RecoverAI - Athlete Injury Recovery Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Model-Ensemble-purple.svg" alt="Ensemble">
  <img src="https://img.shields.io/badge/Accuracy-95.8%25-brightgreen.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/Precision-100%25-brightgreen.svg" alt="Precision">
</p>

<p align="center">
  <strong>AI-powered injury recovery prediction system using advanced ensemble machine learning</strong>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Recovery Outlook Guide](#-recovery-outlook-guide)
- [Contributing](#-contributing)

---

## 🎯 Overview

RecoverAI is a cutting-edge web application that predicts athlete injury recovery timelines using advanced machine learning. The system analyzes 15+ risk factors and provides:

- **Accurate recovery timeline predictions**
- **Personalized rehabilitation recommendations**
- **Risk factor analysis**
- **Confidence scores for predictions**

### Key Achievements

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 95.83% |
| **Precision** | 100% |
| **Recall** | 87% |
| **F1-Score** | 93% |
| **AUC-ROC** | 93% |

---

## ✨ Features

### Core Features

- 🤖 **Ensemble ML Model**: Combines Gradient Boosting, Random Forest, SVM, and Logistic Regression
- 📊 **30 Engineered Features**: Advanced feature engineering for maximum accuracy
- ⚖️ **SMOTE Balancing**: Handles class imbalance for fair predictions
- 🎯 **Recovery Timeline**: Estimates days to recovery (3-42 days range)
- 💡 **Smart Recommendations**: Context-aware rehabilitation advice
- 📱 **Responsive UI**: Beautiful dark theme with emerald accents

### Recovery Outlooks

| Outlook | Risk Range | Timeline |
|---------|------------|----------|
| 🟢 Excellent | < 25% | 3-5 days |
| 🟢 Good | 25-50% | 5-10 days |
| 🟡 Moderate | 50-75% | 10-21 days |
| 🔴 Extended | > 75% | 21-42 days |

---

## 📈 Model Performance

### Algorithm Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Gradient Boosting** | **95.83%** | **100%** | **86.84%** | **92.96%** |
| Random Forest | 95.00% | 97.06% | 86.84% | 91.67% |
| Ensemble (Voting) | 93.33% | 100% | 78.95% | 88.24% |
| Optimized SVM | 88.33% | 87.50% | 73.68% | 80.00% |
| Bagging SVM | 85.83% | 80.00% | 73.68% | 76.71% |
| Logistic Regression | 85.00% | 81.25% | 68.42% | 74.29% |

### Feature Engineering

The model uses 30 features including:

**Original Features (15)**
- Personal: Age, Gender, Height, Weight, BMI
- Training: Frequency, Duration, Warm-up, Intensity
- Physical: Flexibility, Muscle Asymmetry, Injury History
- Recovery: Sleep Hours, Recovery Time, Stress Level

**Engineered Features (15)**
- Training_Load, Recovery_Ratio, Sleep_Recovery_Score
- Risk_Index, Physical_Stress, Warmup_Ratio
- BMI_Age_Factor, Fitness_Score, Training_Efficiency
- Recovery_Need, polynomial features, categorical bins

---

## 🛠️ Technology Stack

### Backend
- **Python 3.9+** - Core language
- **Flask 3.0** - Web framework
- **scikit-learn 1.3.2** - Machine learning
- **imbalanced-learn** - SMOTE implementation
- **pandas** - Data processing

### Frontend
- **HTML5 / CSS3** - Structure & styling
- **JavaScript (ES6+)** - Interactivity
- **Font Awesome 6** - Icons
- **Google Fonts** - Typography (Outfit, Space Mono)

### ML Pipeline
- **Ensemble Learning** - Multiple model combination
- **Gradient Boosting** - Primary classifier
- **Feature Engineering** - 15+ derived features
- **SMOTE-Tomek** - Class balancing
- **SelectKBest** - Feature selection

---

## 📁 Project Structure

```
athlete_injury_prediction/
│
├── app.py                      # Flask application
├── train_model.py              # Model training script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── models/                     # Trained models
│   ├── recovery_prediction_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   └── feature_info.pkl
│
├── static/
│   ├── css/
│   │   └── style.css           # Stylesheet
│   └── js/
│       └── main.js             # JavaScript
│
├── templates/
│   ├── index.html              # Home page
│   └── about.html              # About page
│
└── High_Accuracy_Sport_Injury_Dataset.xlsx
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip

### Quick Start

```bash
# 1. Clone/download the project
cd athlete_injury_prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (if needed)
python train_model.py

# 5. Run the application
python app.py

# 6. Open browser
# Navigate to http://127.0.0.1:5000
```

---

## 📖 Usage

### Making a Prediction

1. **Access the application** at `http://127.0.0.1:5000`

2. **Fill in the assessment form**:
   - Personal information (Age, Gender, Height, Weight)
   - Training details (Frequency, Duration, Intensity)
   - Physical metrics (Flexibility, Muscle Asymmetry)
   - Recovery factors (Sleep, Recovery Time, Stress)

3. **Click "Predict Recovery Timeline"**

4. **Review your results**:
   - Recovery outlook (Excellent/Good/Moderate/Extended)
   - Estimated recovery days
   - Probability breakdown
   - Factor analysis
   - Personalized recommendations

### Input Guidelines

| Field | Range | Description |
|-------|-------|-------------|
| Age | 15-60 | Years |
| Gender | 0/1 | Female/Male |
| Height | 140-220 | Centimeters |
| Weight | 40-150 | Kilograms |
| BMI | 15-45 | Auto-calculated |
| Training Frequency | 1-7 | Days/week |
| Training Duration | 15-300 | Minutes |
| Warm-up Time | 0-60 | Minutes |
| Training Intensity | 1-10 | Self-rated |
| Flexibility Score | 1-10 | Mobility rating |
| Muscle Asymmetry | 0-30 | % difference |
| Injury History | 0-20 | Previous injuries |
| Sleep Hours | 4-12 | Per night |
| Recovery Time | 12-72 | Hours between sessions |
| Stress Level | 1-10 | Mental stress |

---

## 🔌 API Endpoints

### Prediction Endpoint

```http
POST /predict
Content-Type: application/json

{
    "age": 25,
    "gender": 1,
    "height": 175,
    "weight": 70,
    "bmi": 22.9,
    "training_frequency": 5,
    "training_duration": 90,
    "warmup_time": 15,
    "sleep_hours": 7.5,
    "flexibility_score": 6,
    "muscle_asymmetry": 5,
    "recovery_time": 48,
    "injury_history": 2,
    "stress_level": 4,
    "training_intensity": 7
}
```

### Response

```json
{
    "success": true,
    "prediction": 0,
    "extended_recovery_probability": 15.5,
    "quick_recovery_probability": 84.5,
    "recovery_outlook": "Excellent",
    "outlook_class": "excellent",
    "estimated_recovery_days": "3-5 days",
    "model_confidence": 95.2,
    "recommendations": [...],
    "analysis": {
        "training_load_status": "Moderate",
        "sleep_status": "Optimal",
        "flexibility_status": "Average",
        "stress_status": "Low",
        "injury_history_impact": "Moderate"
    }
}
```

### Health Check

```http
GET /api/health

{
    "status": "healthy",
    "model_loaded": true,
    "accuracy": 0.9583,
    "model_name": "Gradient Boosting"
}
```

---

## 📊 Recovery Outlook Guide

### 🟢 Excellent (< 25% risk)
- **Timeline**: 3-5 days
- **Action**: Continue current practices
- **Focus**: Maintain good habits

### 🟢 Good (25-50% risk)
- **Timeline**: 5-10 days
- **Action**: Minor optimizations
- **Focus**: Sleep quality, warm-up routine

### 🟡 Moderate (50-75% risk)
- **Timeline**: 10-21 days
- **Action**: Proactive measures needed
- **Focus**: Reduce training load, improve flexibility

### 🔴 Extended (> 75% risk)
- **Timeline**: 21-42 days
- **Action**: Immediate intervention required
- **Focus**: Consult sports medicine professional

---

## 🤝 Contributing

Contributions welcome! Ideas for improvement:

- [ ] Add more ML models
- [ ] Implement user authentication
- [ ] Add injury history tracking
- [ ] Create mobile app
- [ ] Add visualization dashboard
- [ ] Multi-language support

---

## 📄 License

MIT License - see LICENSE file for details.

---

<p align="center">
  <strong>Built with 💚 for Athletes</strong>
</p>

<p align="center">
  <em>Recover Smarter, Perform Better</em>
</p>
