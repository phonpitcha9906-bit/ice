#!/usr/bin/env python3
"""
predict_plant_growth_xgboost_bayesopt.py
This script provides a high-performance approach to plant growth prediction using XGBoost.
Key features include:
1. Data Preprocessing: Converts descriptive ranges into numerical midpoints.
2. Automated Hyperparameter Tuning: Utilizes Bayesian Optimization to efficiently find the
   optimal XGBoost hyperparameters (learning_rate, max_depth, n_estimators, subsample, etc.)
   for each target variable by minimizing cross-validation error.
3. Feature Importance Analysis: Calculates and displays which environmental factors are most influential.
4. Final Model Training: Trains robust XGBoost models using the optimized hyperparameters on the full dataset.
5. Prediction: Generates final predictions on the unseen test data.
6. Performance Evaluation: Displays model performance metrics and validation results.
7. Output: Displays tuning results, feature importance, and final predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import re
import warnings
warnings.filterwarnings('ignore')

print("=== Plant Growth Prediction using XGBoost with Bayesian Optimization ===")
print("Starting model training and optimization process...\n")

def convert_range_to_midpoint(range_str):
    """
    Convert range string to numerical midpoint
    """
    if pd.isna(range_str):
        return np.nan
    
    # Remove units and extra characters
    clean_str = re.sub(r'[%RHluxppm\s]+', '', str(range_str))
    parts = clean_str.split('-')
    
    if len(parts) == 2:
        try:
            start_val = float(parts[0])
            end_val = float(parts[1])
            return (start_val + end_val) / 2
        except ValueError:
            return np.nan
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    else:
        return np.nan

# --- 1. Load and Preprocess Data ---
print("1. Loading and preprocessing data...")
try:
    df_train = pd.read_excel('datatraining.xlsx')
    df_test = pd.read_excel('datatest.xlsx')
    print(f"   ✓ Data files loaded successfully")
    print(f"   Training data size: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"   Test data size: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
except Exception as e:
    print(f"   ✗ Error loading data files: {e}")
    exit()

# Define columns to convert
cols_to_convert = ['Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration']
print("\n2. Converting feature columns to numerical values...")

for col in cols_to_convert:
    if col in df_train.columns:
        df_train[col] = df_train[col].apply(convert_range_to_midpoint)
        df_test[col] = df_test[col].apply(convert_range_to_midpoint)
        print(f"   ✓ Column \"{col}\" converted successfully")
    else:
        print(f"   ⚠ Warning: Column \"{col}\" not found in data")

# Check for missing values
missing_train = df_train[cols_to_convert].isna().sum().sum()
missing_test = df_test[cols_to_convert].isna().sum().sum()
if missing_train > 0 or missing_test > 0:
    print(f"   ⚠ Warning: Found {missing_train} missing values in training data, {missing_test} in test data")
    # Fill missing values with median
    for col in cols_to_convert:
        median_val = df_train[col].median()
        df_train[col].fillna(median_val, inplace=True)
        df_test[col].fillna(median_val, inplace=True)

# --- 2. Define Features (X) and Targets (y) ---
print("\n3. Preparing feature matrices and target variables...")
X_train = df_train[cols_to_convert].values
X_test = df_test[cols_to_convert].values
feature_names = cols_to_convert

# Extract target variables
target_vars = ['LeafLength', 'LeafArea', 'StemGirth']
for target in target_vars:
    if target not in df_train.columns:
        print(f"   ✗ Error: Target variable \"{target}\" not found in training data")
        exit()

y_train_LeafLength = df_train['LeafLength'].values
y_train_LeafArea = df_train['LeafArea'].values
y_train_StemGirth = df_train['StemGirth'].values

print(f"   ✓ Feature matrix dimensions: {X_train.shape}")
print(f"   ✓ Test matrix dimensions: {X_test.shape}")
print(f"   ✓ Target variables prepared: {', '.join(target_vars)}")

# --- 3. Define Hyperparameter Optimization ---
print("\n4. Setting up Bayesian Optimization for hyperparameter tuning...")

# Define hyperparameter search space for XGBoost
space = [
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth'),
    Integer(50, 500, name='n_estimators'),
    Real(0.6, 1.0, name='subsample'),
    Real(0.6, 1.0, name='colsample_bytree'),
    Real(0, 10, name='reg_alpha'),
    Real(0, 10, name='reg_lambda')
]

def optimize_xgboost(X, y, target_name):
    """
    Optimize XGBoost hyperparameters using Bayesian Optimization
    """
    print(f"   Optimizing hyperparameters for {target_name}...")
    
    @use_named_args(space)
    def objective(**params):
        # Create XGBoost regressor with current parameters
        model = xgb.XGBRegressor(
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=42,
            verbosity=0
        )
        
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                  scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Return negative MSE (since we want to minimize)
        return -cv_scores.mean()
    
    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=30, random_state=42)
    
    return result

# --- 4. Hyperparameter Tuning for Each Target Variable ---
print("\n5. Performing hyperparameter optimization...")

# Optimize for LeafLength
results_LL = optimize_xgboost(X_train, y_train_LeafLength, 'LeafLength')
best_params_LL = {
    'learning_rate': results_LL.x[0],
    'max_depth': results_LL.x[1],
    'n_estimators': results_LL.x[2],
    'subsample': results_LL.x[3],
    'colsample_bytree': results_LL.x[4],
    'reg_alpha': results_LL.x[5],
    'reg_lambda': results_LL.x[6]
}

# Optimize for LeafArea
results_LA = optimize_xgboost(X_train, y_train_LeafArea, 'LeafArea')
best_params_LA = {
    'learning_rate': results_LA.x[0],
    'max_depth': results_LA.x[1],
    'n_estimators': results_LA.x[2],
    'subsample': results_LA.x[3],
    'colsample_bytree': results_LA.x[4],
    'reg_alpha': results_LA.x[5],
    'reg_lambda': results_LA.x[6]
}

# Optimize for StemGirth
results_SG = optimize_xgboost(X_train, y_train_StemGirth, 'StemGirth')
best_params_SG = {
    'learning_rate': results_SG.x[0],
    'max_depth': results_SG.x[1],
    'n_estimators': results_SG.x[2],
    'subsample': results_SG.x[3],
    'colsample_bytree': results_SG.x[4],
    'reg_alpha': results_SG.x[5],
    'reg_lambda': results_SG.x[6]
}

# Display optimization results
print("\n6. Hyperparameter optimization results:")
print(f"   LeafLength - lr: {best_params_LL['learning_rate']:.3f}, depth: {best_params_LL['max_depth']}, "
      f"n_est: {best_params_LL['n_estimators']}, subsample: {best_params_LL['subsample']:.3f} (CV Loss: {results_LL.fun:.4f})")
print(f"   LeafArea   - lr: {best_params_LA['learning_rate']:.3f}, depth: {best_params_LA['max_depth']}, "
      f"n_est: {best_params_LA['n_estimators']}, subsample: {best_params_LA['subsample']:.3f} (CV Loss: {results_LA.fun:.4f})")
print(f"   StemGirth  - lr: {best_params_SG['learning_rate']:.3f}, depth: {best_params_SG['max_depth']}, "
      f"n_est: {best_params_SG['n_estimators']}, subsample: {best_params_SG['subsample']:.3f} (CV Loss: {results_SG.fun:.4f})")

# --- 5. Train Final Models with Optimal Hyperparameters ---
print("\n7. Training final XGBoost models with optimal hyperparameters...")

def train_final_model(X, y, params, feature_names, target_name):
    """
    Train final XGBoost model and calculate feature importance
    """
    # Train XGBoost with optimal parameters
    model = xgb.XGBRegressor(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42,
        verbosity=0
    )
    
    model.fit(X, y)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Normalize importance scores to percentages
    importance_pct = (importance / importance.sum()) * 100
    
    # Display feature importance
    print(f"   Feature importance for {target_name}:")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_pct
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"     {row['Feature']}: {row['Importance']:.2f}%")
    print()
    
    return model, importance_pct

# Train final models
model_LeafLength, importance_LL = train_final_model(X_train, y_train_LeafLength, best_params_LL, feature_names, 'LeafLength')
model_LeafArea, importance_LA = train_final_model(X_train, y_train_LeafArea, best_params_LA, feature_names, 'LeafArea')
model_StemGirth, importance_SG = train_final_model(X_train, y_train_StemGirth, best_params_SG, feature_names, 'StemGirth')

# --- 6. Model Performance Evaluation ---
print("8. Evaluating model performance on training data...")

def evaluate_model(model, X, y, target_name):
    """
    Evaluate model performance
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"   {target_name} Performance:")
    print(f"     RMSE: {rmse:.4f}")
    print(f"     MAE:  {mae:.4f}")
    print(f"     R²:   {r2:.4f}")
    print()

evaluate_model(model_LeafLength, X_train, y_train_LeafLength, 'LeafLength')
evaluate_model(model_LeafArea, X_train, y_train_LeafArea, 'LeafArea')
evaluate_model(model_StemGirth, X_train, y_train_StemGirth, 'StemGirth')

# --- 7. Generate Final Predictions ---
print("9. Generating predictions for test data...")

predictions_LeafLength = model_LeafLength.predict(X_test)
predictions_LeafArea = model_LeafArea.predict(X_test)
predictions_StemGirth = model_StemGirth.predict(X_test)

print(f"   ✓ Predictions generated for {len(predictions_LeafLength)} test samples")

# --- 8. Display and Save Results ---
print("\n10. Final Results:")

# Create results dataframe
results_df = pd.DataFrame({
    'PredictedLeafLength': predictions_LeafLength,
    'PredictedLeafArea': predictions_LeafArea,
    'PredictedStemGirth': predictions_StemGirth
})

# Display results
print("=== Final Predictions (XGBoost with Bayesian Optimization) ===")
print(results_df)

# Save results to CSV file
try:
    results_df.to_csv('plant_growth_predictions_xgboost_bayesopt.csv', index=False)
    print('\n   ✓ Results saved to "plant_growth_predictions_xgboost_bayesopt.csv"')
except Exception as e:
    print(f'\n   ⚠ Could not save results to CSV file: {e}')

# --- 9. Summary ---
print("\n=== Model Training Summary ===")
print("Algorithm: XGBoost with Bayesian Optimization")
print(f"Features used: {', '.join(feature_names)}")
print(f"Target variables: {', '.join(target_vars)}")
print("Optimization method: Bayesian Optimization with Cross-Validation")
print(f"Number of test predictions: {len(results_df)}")
print("\nProcess completed successfully!")