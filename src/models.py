import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

from src.preprocessing import get_preprocessor

def train_logistic_regression(
  X_train: pd.DataFrame, 
  y_train: pd.Series, 
  categorical_features: list, 
  numerical_features: list
) -> Pipeline:
  """Train Logistic Regression with preprocessing pipeline."""
  pipeline = Pipeline(steps=[
    ("preprocessor", get_preprocessor(categorical_features, numerical_features)),
    ("model", LogisticRegression(max_iter=1000))
  ])
  pipeline.fit(X_train, y_train)
  return pipeline

def train_decision_tree(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    categorical_features: list, 
    numerical_features: list
) -> Pipeline:
    """Train Decision Tree with preprocessing pipeline."""
    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor(categorical_features, numerical_features)),
        ("model", DecisionTreeClassifier(max_depth=6, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_catboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    categorical_features: list
) -> CatBoostClassifier:
    """Train CatBoost model (handles categorical features natively)."""
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train, cat_features=categorical_features)
    return model
