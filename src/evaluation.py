import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

def cross_validate_model(train_func, X: pd.DataFrame, y: pd.Series, name: str, cv: int = 5, **kwargs):
  cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
  scores = []

  for train_idx, val_idx in cv_strategy.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = train_func(X_train, y_train, **kwargs)
    y_proba = model.predict_proba(X_val)[:, 1]
    scores.append(roc_auc_score(y_val, y_proba))

  return {
    "Model": name,
    "Mean ROC-AUC": round(np.mean(scores), 4),
    "Std": round(np.std(scores), 4)
  }