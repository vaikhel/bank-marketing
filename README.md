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

### Test Set Performance

| Model               | Accuracy | F1    | ROC-AUC |
|---------------------|----------|-------|---------|
| Logistic Regression | 0.7548   | 0.3732| 0.7722  |
| Decision Tree       | 0.8232   | 0.4196| 0.7461  |
| CatBoost            | 0.8298   | 0.4541| 0.7912  |

### Cross-Validation (5-fold, ROC-AUC)

| Model               | Mean ROC-AUC | Std   |
|---------------------|--------------|-------|
| Logistic Regression | 0.7664       | 0.0081|
| Decision Tree       | 0.7474       | 0.0107|
| CatBoost            | 0.7893       | 0.0079|

All models use class balancing (`class_weight="balanced"` / `auto_class_weights="Balanced"`) to better handle the imbalanced dataset (~11.7% positive class). This improved F1 scores at the cost of lower Accuracy.

## Key Findings

- CatBoost achieved the highest F1 (0.454) and competitive ROC-AUC.
- Class balancing helped all models better identify clients who subscribe to a term deposit.
- The most important features according to CatBoost: `month`, `day`, `poutcome`, `contact`, `balance`.
- Low standard deviation in cross-validation indicates stable performance.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---