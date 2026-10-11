import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    """Calculate Accuracy, F1 and ROC-AUC for a model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4)
    }