# Deep Learning–Based Credit Risk Prediction

This repository implements and evaluates **deep learning and machine learning models** for predicting credit risk, based on the **Default of Credit Card Clients dataset (UCI Machine Learning Repository)**.  
Our work benchmarks traditional models against advanced architectures, integrates explainability (SHAP/LIME), and explores hybrid ensemble methods.

---

## 📊 Motivation
Accurate credit risk assessment is critical for financial institutions. Traditional scoring models often fail to capture **non-linear relationships** in customer behavior, leading to:
- Financial losses for institutions  
- Missed opportunities for creditworthy borrowers  
- System inefficiencies and instability  

This project leverages **neural architectures** and **ensemble learning** to enhance prediction accuracy while maintaining interpretability.

---

## 🗂️ Dataset
We use the **Default of Credit Card Clients** dataset (UCI ML Repository):  
- **30,000 records**, 23 attributes  
- Features: demographics, credit history, payment behavior, loan characteristics  
- **Target**: Default (Yes/No)  
- **Distribution**: 22.1% default, 77.9% non-default  

> ⚠️ Raw dataset is **not included** in this repo. Please download from:  
> [UCI ML Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 🏗️ Methodology
### Data Preprocessing
- **Missing value treatment** with imputation  
- **Feature engineering**: debt-to-income, payment-to-debt, credit utilization ratio  
- **Encoding**: one-hot + ordinal  
- **Scaling**: standard normalization  
- **Imbalance handling**: SMOTE, class weights, focal loss  

### Models Implemented
**Baseline ML Models**
- Logistic Regression (L1/L2 regularization)  
- Random Forest (ensemble trees)  
- XGBoost (gradient boosting, tuned)  

**Deep Learning Models**
- **Bi-Directional LSTM (BDLSTM)**: captures sequential bill/payment patterns  
- **Transformer (TabTransformer)**: models categorical–numerical interactions  
- **Multi-Layer Perceptron (MLP)**: tuned fully-connected architecture  

### Explainability
- **SHAP** and **LIME** applied to highlight influential features (e.g., credit limit, payment history)

---

## ⚡ Results (Default of Credit Card Dataset)

| Model            | Accuracy | F1-Score | ROC-AUC |
|------------------|----------|----------|---------|
| Transformer      | 81.4%    | 45.7%    | 76.9%   |
| Bi-LSTM          | 77.0%    | 44.5%    | 74.3%   |
| MLP              | 81.8%    | 47.1%    | 76.7%   |
| XGBoost          | 81.9%    | 46.9%    | **77.7%** |
| Random Forest    | 81.2%    | 45.7%    | 75.0%   |

🔎 **Insights**:
- XGBoost delivered the top ROC-AUC (77.7%) and matched best accuracy.  
- Transformer and MLP offered competitive baselines.  
- Deep models offer extensibility for multi-task/multi-modal risk systems.  
- Explainability confirmed known domain factors (limit balance, payment history).

---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/credit-risk-prediction.git
cd credit-risk-prediction

python -m venv .venv
# Windows
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
