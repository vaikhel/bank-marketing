# Bank Marketing Classification

Predicting whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese bank.

## Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- File used: `bank-full.csv`
- Task: Binary classification
- Target: `y` (yes/no)

The `duration` feature was removed to avoid data leakage, as it is only known after the call.

## Project Structure

```
bank-marketing/
├── data/
├── notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Models

Three models were trained and compared:

- Logistic Regression
- Decision Tree
- CatBoost

## Results

| Model               | Accuracy | F1    | ROC-AUC |
|---------------------|----------|-------|---------|
| Logistic Regression | 0.8933   | 0.2815| 0.7717  |
| Decision Tree       | 0.8919   | 0.2892| 0.7136  |
| CatBoost            | 0.8974   | 0.3911| 0.8034  |

CatBoost showed the best performance. The relatively low F1 scores are expected due to class imbalance (~11.7% positive class).

## Key Findings

- The most important features according to CatBoost: `month`, `day`, `poutcome`, `contact`, `balance`.
- Gradient boosting outperformed simpler models on this task.
- ROC-AUC is a more informative metric than Accuracy given the imbalance.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---