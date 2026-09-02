import os
import joblib
import pandas as pd

_model = None

def load_model():
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
        _model = joblib.load(model_path)
    return _model

def predict_lead(lead_data: dict) -> dict:
    """
    Predict conversion likelihood for a single lead.
    
    lead_data: dict containing keys:
        'source', 'city', 'area', 'property_type', 
        'budget_pkr_lac', 'bedrooms', 'is_overseas', 
        'referred_by_existing_client', 'has_financing_approved'
    """
    model = load_model()
    
    # Convert single dict to DataFrame
    df = pd.DataFrame([lead_data])
    
    # Predict
    pred_class = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    # Human readable label
    if prob >= 0.7:
        label = "High likelihood of conversion"
    elif prob >= 0.4:
        label = "Medium likelihood of conversion"
    else:
        label = "Low likelihood of conversion"
        
    return {
        "prediction": int(pred_class),
        "probability": float(prob),
        "label": label
    }
