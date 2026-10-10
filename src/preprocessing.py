import pandas as pd

def load_data(path: str = "../data/bank-full.csv") -> pd.DataFrame:
  """Load the Bank Marketing dataset and drop the duration feature (data leakage)."""
  df = pd.read_csv(path, sep=";")
  df = df.drop(columns=["duration"])
  return df