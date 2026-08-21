# Online Payments Fraud Detection

This project uses machine learning to detect fraudulent online payment transactions. Using the PaySim dataset, a Decision Tree Classifier was trained to identify fraudulent behavior based on transaction types, amounts, and account balance discrepancies.

This project was completed as part of the TUMO Labs Station Project.

## Setup / How to Run

1. Clone this repository or download the notebook file.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download) and place the CSV file in the same folder as the notebook.
3. Install the required libraries:
   ```bash
   pip install pandas numpy plotly scikit-learn
   ```
4. Open the notebook in Jupyter and run all cells in order.

## Data Architecture & Exploration

The dataset used is **[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)**, a synthetic dataset simulating mobile money transactions based on financial logs from an African mobile money service.

* **Dataset Size:** 6,362,620 transactions | 11 columns
* **Class Imbalance:** 8,213 fraudulent transactions (**0.13%** of total)

### Key Findings

* **Fraud-Prone Categories:** Fraud occurs **exclusively** in `CASH_OUT` and `TRANSFER` transactions. `PAYMENT`, `CASH_IN`, and `DEBIT` have a **0%** fraud rate.
* **Transaction Size Cap:** Fraudulent transactions carry significantly higher average values (mean: ~$1.47M vs. $178K legitimate) and hit a strict cap of **$10,000,000**.
* **Account Draining:** Fraudulent transactions originate from accounts with high initial balances (~$1.65M average) that are almost completely drained upon transfer.
* **Excluded Features:** `step` (no strong temporal trends) and merchant prefix identifiers (`nameDest` starting with 'M', perfectly correlated with zero-fraud `PAYMENT` types) were excluded.

## Feature Engineering & Selection

To optimize performance and eliminate noise, training was restricted strictly to `CASH_OUT` and `TRANSFER` transactions.

### Feature Mapping

| Feature | Type | Reason & Insight |
| :--- | :--- | :--- |
| `type` | Encoded (`1`, `2`) | Filtered to CASH_OUT and TRANSFER only |
| `amount` | Numerical | High transaction amounts strongly correlate with fraud |
| `oldbalanceOrg` | Numerical | High starting balance prior to account drain |
| `newbalanceOrig` | Numerical | Near-zero ending balance following fraud |
| `oldbalanceDest` | Numerical | Initial balance of recipient |
| `newbalanceDest` | Numerical | Updated balance of recipient |
| `errorBalanceOrig` | **Engineered** | `newbalanceOrig + amount - oldbalanceOrg` (Sender balance discrepancy) |
| `errorBalanceDest` | **Engineered** | `oldbalanceDest + amount - newbalanceDest` (Recipient balance discrepancy) |

## Model Pipeline

A **Decision Tree Classifier** was trained using an 80/20 stratified split (`stratify=y`) to preserve the fraud ratio in both sets, given the severe class imbalance.

```python
DecisionTreeClassifier(max_depth=10, random_state=42)
```

* `max_depth=10` limits tree depth to avoid overfitting.
* `random_state=42` ensures reproducible results.

## Evaluation

**Accuracy:** 100.00% — misleading on its own given the imbalance, so precision/recall matter more here.

**Confusion Matrix (test set, 554,082 transactions):**

|  | Predicted: Non-Fraud | Predicted: Fraud |
| :--- | :--- | :--- |
| **Actual: Non-Fraud** | 552,436 | 3 |
| **Actual: Fraud** | 6 | 1,637 |

**Classification Report (Fraud class):** Precision 1.00, Recall 1.00, F1-score 1.00 — out of 1,643 real fraud cases, the model caught 1,637, missing only 6, with just 3 false alarms.

**Example prediction:** for a sample test transaction (type=CASH_OUT, amount=$226,596, oldbalanceOrg=$0), the model correctly predicted `isFraud = 0`, matching the actual label.

## Conclusion

* **Domain-Driven Feature Engineering Works:** Creating explicit balance discrepancy metrics (`errorBalanceOrig`) increased model precision and recall from ~0.91 to near-perfect performance (~0.996).
* **Targeted Scope Reduces Noise:** Filtering the training scope strictly to `CASH_OUT` and `TRANSFER` transaction types eliminated false signals from non-risk types (`PAYMENT`, `DEBIT`, `CASH_IN`).
* **Operational Viability:** The model achieves an optimal balance between catching fraud (1,637 / 1,643 detected) and minimizing customer friction (only 3 false positives out of 552k+ normal transactions).