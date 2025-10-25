import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset diabetes
print("Loading Diabetes Dataset...")
df = pd.read_csv("datasets.csv")
print("Dataset loaded successfully!")

# Data Exploration
print("\n" + "="*50)
print("DATASET EXPLORATION")
print("="*50)

print("Dataset Info:")
df.info()
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# Check target variable distribution
target_column = 'Diabetes_012'
print(f"\nTarget Variable Distribution:")
print(df[target_column].value_counts())
print(f"\nTarget Variable Percentage:")
print(df[target_column].value_counts(normalize=True) * 100)

# Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Identify feature columns (excluding target variable)
feature_columns = [col for col in df.columns if col != target_column]
print(f"Feature columns: {feature_columns}")

# Separate features and target
X = df[feature_columns]
y = df[target_column]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

print(f"\nFeatures after scaling:")
print(X_scaled.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData Split Results:")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# Model Training
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

print("Training Random Forest Classifier...")
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)

print("Model training completed!")

# Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Calculate metrics for multiclass classification
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
print(f"\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Model Inference Example
print("\n" + "="*50)
print("MODEL INFERENCE EXAMPLE")
print("="*50)

# Example prediction
print("Example: Predicting diabetes for a new patient...")
print("Enter patient data (or use sample data):")

# Sample patient data (you can modify these values)
sample_patient = {
    'HighBP': 1.0,
    'HighChol': 0.0,
    'CholCheck': 1.0,
    'BMI': 25.5,
    'Smoker': 0.0,
    'Stroke': 0.0,
    'HeartDiseaseorAttack': 0.0,
    'PhysActivity': 1.0,
    'Fruits': 1.0,
    'Veggies': 1.0,
    'HvyAlcoholConsump': 0.0,
    'AnyHealthcare': 1.0,
    'NoDocbcCost': 0.0,
    'GenHlth': 3.0,
    'MentHlth': 5.0,
    'PhysHlth': 5.0,
    'DiffWalk': 0.0,
    'Sex': 0.0,
    'Age': 9.0,
    'Education': 4.0,
    'Income': 3.0
}

print(f"Sample patient data: {sample_patient}")

# Create DataFrame for prediction
patient_df = pd.DataFrame([sample_patient])
patient_scaled = scaler.transform(patient_df)
patient_scaled_df = pd.DataFrame(patient_scaled, columns=feature_columns)

# Make prediction
prediction = rf_classifier.predict(patient_scaled_df)[0]
prediction_proba = rf_classifier.predict_proba(patient_scaled_df)[0]

print(f"\nPrediction Result:")
print(f"Diabetes Risk: {'HIGH (Diabetic)' if prediction == 1 else 'LOW (Non-Diabetic)'}")
print(f"Probability of Diabetes: {prediction_proba[1]:.4f}")
print(f"Probability of No Diabetes: {prediction_proba[0]:.4f}")

# Export Model and Scaler
print("\n" + "="*50)
print("EXPORTING MODEL")
print("="*50)

import joblib

# Export the trained model
model_filename = 'diabetes_rf_model.joblib'
joblib.dump(rf_classifier, model_filename)
print(f"Model exported successfully as {model_filename}")

# Export the scaler
scaler_filename = 'diabetes_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Scaler exported successfully as {scaler_filename}")

print("\nModel and scaler exported successfully!")
print("You can now use these files for inference in your Streamlit app.")

# Visualization
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# 2. Feature Importance
feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[0,1])
axes[0,1].set_title('Feature Importance')
axes[0,1].set_xlabel('Importance')

# 3. Target Distribution
y.value_counts().plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Target Variable Distribution')
axes[1,0].set_xlabel('Outcome')
axes[1,0].set_ylabel('Count')

# 4. Prediction Probabilities Distribution
axes[1,1].hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='red')
axes[1,1].set_title('Distribution of Diabetes Probabilities')
axes[1,1].set_xlabel('Probability of Diabetes')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('diabetes_model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'diabetes_model_analysis.png'")
print("\nTraining completed successfully!")
