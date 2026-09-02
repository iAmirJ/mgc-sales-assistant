import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import os

from .preprocessing import get_preprocessing_pipeline, CityCleaner

def train_model():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'leads.csv')
    df = pd.read_csv(data_path)

    # Drop duplicates by crm_record_hash to prevent leakage across train/test
    # This prevents the same exact lead (if recorded twice) from appearing in both train and test.
    df = df.drop_duplicates(subset=['crm_record_hash'])

    # Define features (LEAKAGE SAFE)
    # Excluded: first_response_minutes, calls_made, total_call_seconds, whatsapp_replies, 
    # site_visits, agent_experience_years, token_amount_received_pkr
    features = [
        'source', 'city', 'area', 'property_type', 'budget_pkr_lac', 
        'bedrooms', 'is_overseas', 'referred_by_existing_client', 'has_financing_approved'
    ]
    target = 'converted'

    X = df[features]
    y = df[target]

    # Temporal split could be used, but since we don't know the exact nature of the timeframe,
    # and the prompt suggests stratified split if data doesn't strictly require temporal,
    # let's use stratified train_test_split to maintain class balance.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # We will use Gradient Boosting as it handles mixed feature types well and generally outperforms LR.
    model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=4)

    # Full Pipeline
    pipeline = Pipeline(steps=[
        ('city_cleaner', CityCleaner()),
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', model)
    ])

    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Predict on test
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc_average_precision': average_precision_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print("--- Model Evaluation ---")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}:\n{v}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_model()
