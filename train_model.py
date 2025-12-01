"""
Athlete Injury Recovery Prediction Model - Advanced Training Script
Using Optimized SVM with Feature Engineering and Ensemble Techniques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("=" * 70)
    print("   ATHLETE INJURY RECOVERY PREDICTION MODEL - ADVANCED TRAINING")
    print("=" * 70)
    
    df = pd.read_excel('High_Accuracy_Sport_Injury_Dataset.xlsx')
    print(f"\n📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Rename target for clarity (Recovery prediction context)
    # 0 = Quick Recovery Expected, 1 = Extended Recovery Needed
    
    X = df.drop('Injury_Risk', axis=1)
    y = df['Injury_Risk']
    
    print(f"\n🎯 Target: Recovery Risk Prediction")
    print(f"   Quick Recovery (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Extended Recovery (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y, list(X.columns)

def engineer_features(X):
    """Create advanced feature engineering"""
    print("\n🔧 Engineering advanced features...")
    
    X_eng = X.copy()
    
    # Interaction features
    X_eng['Training_Load'] = X_eng['Training_Frequency'] * X_eng['Training_Duration'] * X_eng['Training_Intensity'] / 100
    X_eng['Recovery_Ratio'] = X_eng['Recovery_Time'] / (X_eng['Training_Duration'] + 1)
    X_eng['Sleep_Recovery_Score'] = X_eng['Sleep_Hours'] * X_eng['Recovery_Time'] / 100
    X_eng['Risk_Index'] = (X_eng['Injury_History'] * X_eng['Training_Intensity']) / (X_eng['Flexibility_Score'] + 1)
    X_eng['Physical_Stress'] = X_eng['Stress_Level'] * X_eng['Muscle_Asymmetry'] / 10
    X_eng['Warmup_Ratio'] = X_eng['Warmup_Time'] / (X_eng['Training_Duration'] + 1)
    X_eng['BMI_Age_Factor'] = X_eng['BMI'] * X_eng['Age'] / 100
    X_eng['Fitness_Score'] = (X_eng['Flexibility_Score'] * X_eng['Sleep_Hours']) / (X_eng['Stress_Level'] + 1)
    X_eng['Training_Efficiency'] = X_eng['Training_Intensity'] / (X_eng['Training_Frequency'] + 1)
    X_eng['Recovery_Need'] = X_eng['Training_Load'] / (X_eng['Sleep_Hours'] + 1)
    
    # Polynomial features for top correlated features
    X_eng['Injury_History_Sq'] = X_eng['Injury_History'] ** 2
    X_eng['Training_Intensity_Sq'] = X_eng['Training_Intensity'] ** 2
    X_eng['Sleep_Hours_Sq'] = X_eng['Sleep_Hours'] ** 2
    
    # Categorical binning
    X_eng['Age_Group'] = pd.cut(X_eng['Age'], bins=[0, 25, 30, 35, 100], labels=[0, 1, 2, 3]).astype(int)
    X_eng['BMI_Category'] = pd.cut(X_eng['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    
    print(f"   Created {X_eng.shape[1] - X.shape[1]} new features")
    print(f"   Total features: {X_eng.shape[1]}")
    
    return X_eng

def train_optimized_model(X, y, feature_names):
    """Train optimized model with multiple techniques"""
    print("\n" + "=" * 70)
    print("   MODEL TRAINING WITH ADVANCED OPTIMIZATION")
    print("=" * 70)
    
    # Feature engineering
    X_eng = engineer_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Data Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Handle class imbalance with SMOTE
    print("\n⚖️ Applying SMOTE for class balancing...")
    smote = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"   Balanced training set: {X_train_balanced.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    print("\n🎯 Selecting best features...")
    selector = SelectKBest(f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_features = [X_eng.columns[i] for i in selector.get_support(indices=True)]
    print(f"   Selected {len(selected_features)} best features")
    
    # Optimized SVM with extensive grid search
    print("\n🔍 Optimizing SVM hyperparameters...")
    svm_param_grid = {
        'C': [0.1, 1, 10, 50, 100, 200],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly'],
        'degree': [2, 3, 4],
        'class_weight': ['balanced', None]
    }
    
    svm = SVC(probability=True, random_state=42)
    
    # Use stratified k-fold
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        svm, svm_param_grid, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=0, refit=True
    )
    grid_search.fit(X_train_selected, y_train_balanced)
    
    best_svm = grid_search.best_estimator_
    print(f"\n✅ Best SVM Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   - {param}: {value}")
    
    # Create ensemble with multiple classifiers
    print("\n🤖 Building Ensemble Model...")
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        random_state=42
    )
    
    # Logistic Regression
    lr = LogisticRegression(
        C=10, class_weight='balanced', max_iter=1000, random_state=42
    )
    
    # Bagging SVM
    bagging_svm = BaggingClassifier(
        estimator=SVC(probability=True, kernel='rbf', C=100, gamma='scale', random_state=42),
        n_estimators=10, random_state=42, n_jobs=-1
    )
    
    # Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('svm', best_svm),
            ('rf', rf),
            ('gb', gb),
            ('lr', lr),
            ('bag_svm', bagging_svm)
        ],
        voting='soft',
        weights=[3, 2, 2, 1, 2]  # Give more weight to SVM
    )
    
    print("   Training ensemble model...")
    ensemble.fit(X_train_selected, y_train_balanced)
    
    # Evaluate all models
    print("\n" + "=" * 70)
    print("   MODEL EVALUATION")
    print("=" * 70)
    
    models = {
        'Optimized SVM': best_svm,
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Logistic Regression': lr,
        'Bagging SVM': bagging_svm,
        'Ensemble (Voting)': ensemble
    }
    
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        if name != 'Ensemble (Voting)':
            model.fit(X_train_selected, y_train_balanced)
        
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_selected, y_train_balanced, cv=5, scoring='accuracy')
        
        print(f"\n📊 {name}:")
        print(f"   Accuracy:  {acc*100:.2f}%")
        print(f"   Precision: {prec*100:.2f}%")
        print(f"   Recall:    {rec*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        print(f"   AUC-ROC:   {auc*100:.2f}%")
        print(f"   CV Score:  {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
    
    print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")
    
    # Final evaluation of best model
    y_pred_final = best_model.predict(X_test_selected)
    
    print("\n📋 Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_final, 
                                target_names=['Quick Recovery', 'Extended Recovery']))
    
    print("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"   [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"    [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    return best_model, scaler, selector, X_eng.columns.tolist(), best_accuracy, best_model_name

def save_model(model, scaler, selector, feature_names, accuracy, model_name):
    """Save all model components"""
    print("\n" + "=" * 70)
    print("   SAVING MODEL")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/recovery_prediction_model.pkl')
    print("\n✅ Model saved: models/recovery_prediction_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Scaler saved: models/scaler.pkl")
    
    # Save feature selector
    joblib.dump(selector, 'models/selector.pkl')
    print("✅ Selector saved: models/selector.pkl")
    
    # Save feature info
    feature_info = {
        'original_features': feature_names[:15],
        'all_features': feature_names,
        'accuracy': accuracy,
        'model_name': model_name
    }
    joblib.dump(feature_info, 'models/feature_info.pkl')
    print("✅ Feature info saved: models/feature_info.pkl")
    
    print("\n" + "=" * 70)
    print(f"   ✅ TRAINING COMPLETE - {model_name}: {accuracy*100:.2f}% ACCURACY")
    print("=" * 70)

def main():
    # Install required package
    import subprocess
    subprocess.run(['pip', 'install', 'imbalanced-learn', '--break-system-packages', '-q'], 
                   capture_output=True)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Train optimized model
    model, scaler, selector, all_features, accuracy, model_name = train_optimized_model(X, y, feature_names)
    
    # Save model
    save_model(model, scaler, selector, all_features, accuracy, model_name)

if __name__ == "__main__":
    main()
