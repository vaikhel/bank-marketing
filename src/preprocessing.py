import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path: str = "../data/bank-full.csv") -> pd.DataFrame:
  """Load the Bank Marketing dataset and drop the duration feature (data leakage)."""
  df = pd.read_csv(path, sep=";")
  df = df.drop(columns=["duration"])
  return df

def get_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """Create preprocessor for categorical and numerical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )