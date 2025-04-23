# Credit Card Fraud Detection

A machine learning project for identifying fraudulent credit card transactions. This implementation uses advanced data preprocessing, feature engineering, and ensemble modeling to detect credit card fraud while handling the challenge of highly imbalanced data.

## Overview

This project implements a robust machine learning pipeline to identify fraudulent credit card transactions. The system uses various techniques to address the imbalanced nature of fraud data, performs extensive feature engineering, implements multiple classification models, and evaluates their performance.

![Model Comparison](https://github.com/AnMaster15/Credit-Card-Fraud-Detection/blob/main/model_comparison.png)

## Features

- **Data preprocessing** with outlier handling and scaling
- **Feature engineering** including:
  - Temporal features (hour of day, day of week, weekend indicators)
  - Amount-based features (normalized amounts, z-scores per time period)
  - Transaction frequency features for user profiles
  - Interaction features from principal components
- **Class imbalance handling** with:
  - SMOTE oversampling
  - Random undersampling
  - SMOTETomek combined approach
  - Cost-sensitive learning
- **Multiple models** including:
  - XGBoost
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - Ensemble methods
- **Model optimization** with:
  - Hyperparameter tuning
  - Threshold optimization
- **Comprehensive evaluation metrics**:
  - Precision-Recall AUC
  - ROC AUC
  - F1 score
  - Custom business metrics

## Getting Started

### Prerequisites

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib
```

### Installation

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Dataset

The project uses a credit card transaction dataset with the following characteristics:
- Features include anonymized PCA components (V1-V28), transaction amount, and time
- Highly imbalanced class distribution (fraud transactions << legitimate transactions)
- Binary classification task (Class: 0 = Normal, 1 = Fraud)

Place your dataset in the project directory as `creditcard.csv`.

## Usage

Run the main script to train and evaluate the models:

```python
python fraud_detection.py
```

### Inference Pipeline

The project includes a complete inference pipeline for making predictions on new transactions:

```python
# Load the trained model and metadata
model = joblib.load('final_model_VotingEnsemble_SMOTE.pkl')
scaler = joblib.load('scaler.pkl')
feature_metadata = joblib.load('feature_metadata.pkl')

# Make predictions on new transactions
predictions = predict_fraud(new_transactions, model, scaler, feature_metadata)
```

## Results

The system achieved excellent performance metrics:
- Precision-Recall AUC: 0.95+
- ROC AUC: 0.98+
- F1 score (with optimal threshold): 0.86+

### Feature Importance

![Feature Importance](https://github.com/AnMaster15/Credit-Card-Fraud-Detection/blob/main/feature_importance%20(1).png)

The most discriminative features for fraud detection were:
1. V17
2. V14
3. V12
4. V10
5. Amount-related features

### Threshold Analysis

![Threshold Analysis](https://github.com/AnMaster15/Credit-Card-Fraud-Detection/blob/main/threshold_analysis.png)

## Model Pipeline

1. **Data Loading & Exploration**: Analyze distributions and identify patterns
2. **Data Preprocessing**: Handle missing values and scale features
3. **Feature Engineering**: Create temporal, amount-based, and interaction features  
4. **Model Selection**: Train and evaluate multiple classification algorithms
5. **Class Imbalance Handling**: Apply sampling techniques to balance training data
6. **Hyperparameter Tuning**: Optimize model parameters via cross-validation
7. **Threshold Optimization**: Fine-tune decision threshold for business requirements
8. **Evaluation**: Comprehensive metrics focusing on high precision and recall
9. **Inference**: Production-ready pipeline for real-time fraud detection

## Future Improvements

- Implement real-time streaming data processing
- Add anomaly detection models for comparison
- Develop API for real-time fraud detection
- Incorporate additional transaction metadata
- Deploy as a microservice with monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset providers
- The scikit-learn, XGBoost, and imbalanced-learn communities
