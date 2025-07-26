# Plant Growth Prediction with XGBoost and Bayesian Optimization

This repository contains a Python implementation of plant growth prediction using XGBoost with Bayesian Optimization, converted from the original MATLAB Decision Tree implementation.

## Key Features

1. **XGBoost Regression**: Advanced gradient boosting algorithm for better prediction accuracy
2. **Bayesian Optimization**: Efficient hyperparameter tuning using Gaussian Process optimization
3. **Feature Importance Analysis**: Built-in XGBoost feature importance calculation
4. **Cross-Validation**: 5-fold cross-validation for robust model evaluation
5. **Multi-Target Prediction**: Predicts LeafLength, LeafArea, and StemGirth simultaneously

## Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Required Python Packages:
- pandas >= 1.3.0
- numpy >= 1.21.0
- xgboost >= 1.6.0
- scikit-learn >= 1.0.0
- scikit-optimize >= 0.9.0
- openpyxl >= 3.0.7
- xlrd >= 2.0.1

## Usage

1. Ensure your data files are in the same directory:
   - `datatraining.xlsx`: Training dataset
   - `datatest.xlsx`: Test dataset

2. Run the script:
   ```bash
   python predict_plant_growth_xgboost_bayesopt.py
   ```

3. The script will output:
   - Hyperparameter optimization results
   - Feature importance for each target variable
   - Model performance metrics
   - Final predictions saved to `plant_growth_predictions_xgboost_bayesopt.csv`

## Key Differences from MATLAB Decision Tree Version

### Algorithm Changes:
- **XGBoost** instead of Decision Trees for better performance and accuracy
- **Gradient Boosting** ensemble method vs single tree approach
- **Built-in regularization** (L1 and L2) to prevent overfitting

### Hyperparameters Optimized:
- `learning_rate`: Controls the contribution of each tree (0.01-0.3)
- `max_depth`: Maximum depth of trees (3-10)
- `n_estimators`: Number of boosting rounds (50-500)
- `subsample`: Fraction of samples used for training each tree (0.6-1.0)
- `colsample_bytree`: Fraction of features used for training each tree (0.6-1.0)
- `reg_alpha`: L1 regularization (0-10)
- `reg_lambda`: L2 regularization (0-10)

### Performance Improvements:
- **Better Accuracy**: XGBoost typically outperforms single Decision Trees
- **Robustness**: Built-in regularization and ensemble approach reduce overfitting
- **Feature Importance**: More reliable importance scores using gain-based metrics
- **Cross-Platform**: Python implementation works on any platform

## Expected Output

The script provides detailed output including:

1. **Data Loading and Preprocessing Status**
2. **Hyperparameter Optimization Results** for each target variable
3. **Feature Importance Rankings** for environmental factors
4. **Model Performance Metrics** (RMSE, MAE, R²)
5. **Final Predictions** saved to CSV file

## Data Format

The script expects Excel files with the following columns:
- **Features**: Moisture, Humidity, LightIntensity, CO2Concentration (with range strings like "60-80%")
- **Targets**: LeafLength, LeafArea, StemGirth (numerical values)

## Performance Notes

- Bayesian optimization runs 30 iterations for each target variable
- Total runtime is typically 2-5 minutes depending on data size
- XGBoost automatically uses multiple CPU cores for parallel processing

## Troubleshooting

- Ensure all required packages are installed
- Check that Excel files are in the correct format
- Verify that all target columns exist in the training data
- Missing values are automatically filled with median values
