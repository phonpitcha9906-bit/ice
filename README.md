# Plant Growth Prediction - Fixed MATLAB Implementation

This repository contains a fixed version of the plant growth prediction MATLAB script that resolves the `optimizableVariable` error.

## Problem Solved

The original script was trying to use `optimizableVariable` and Bayesian optimization functions that require the Statistics and Machine Learning Toolbox, which may not be available or properly licensed in all MATLAB installations.

## Solution

I've created an alternative implementation that replaces Bayesian optimization with **Grid Search**, which uses only base MATLAB functions and the standard TreeBagger (Random Forest) implementation.

## Files

1. **`predict_plant_growth_rf_alternative.m`** - The main script with grid search optimization
2. **`create_sample_data.m`** - Script to generate sample training and test data files
3. **`README.md`** - This documentation file

## Key Changes Made

1. **Replaced Bayesian Optimization with Grid Search**:
   - Removed dependency on `optimizableVariable` and `bayesopt`
   - Implemented custom grid search function using nested loops
   - Uses the same OOB (Out-of-Bag) error metric for optimization

2. **Enhanced Error Handling**:
   - Added try-catch blocks for data loading
   - Graceful handling when test data files don't contain actual values
   - Progress indicators during grid search

3. **Maintained All Original Features**:
   - Data preprocessing (range string to numerical conversion)
   - Feature importance analysis
   - Classification metrics (F1-score, Precision, Recall, Accuracy)
   - Performance evaluation
   - Results export to CSV

## How to Use

### Step 1: Generate Sample Data (Optional)
If you don't have your own data files, run the sample data generator:
```matlab
create_sample_data
```
This will create `datatraining.xlsx` and `datatest.xlsx` files.

### Step 2: Run the Prediction Script
```matlab
predict_plant_growth_rf_alternative
```

## Requirements

- MATLAB (base installation)
- No additional toolboxes required
- Excel files with the following columns:
  - `Moisture` (range strings like "20-30%")
  - `Humidity` (range strings like "40-50% RH")
  - `LightIntensity` (range strings like "1000-2000 lux")
  - `CO2Concentration` (range strings like "300-400 ppm")
  - `LeafLength` (numerical values)
  - `LeafArea` (numerical values)
  - `StemGirth` (numerical values)

## Grid Search Parameters

The script tests these parameter combinations:
- **MinLeafSize**: [1, 5, 10, 15, 20]
- **NumPredictorsToSample**: [1, 2, 3, 4]
- **NumTrees**: [50, 100, 150, 200]

Total combinations: 5 × 4 × 4 = 80 combinations per target variable

## Output

The script provides:
1. **Hyperparameter optimization results** for each target variable
2. **Feature importance analysis**
3. **Model performance metrics** (RMSE, MAE, R²)
4. **Classification evaluation** (F1-score, Precision, Recall, Accuracy)
5. **Final predictions** saved to `plant_growth_predictions_rf_gridsearch.csv`

## Performance

While grid search is more computationally intensive than Bayesian optimization, it:
- Works with any MATLAB installation
- Provides comprehensive parameter space exploration
- Offers reliable and reproducible results
- Shows progress indicators for user feedback

The total runtime depends on your data size but typically completes within a few minutes for datasets of 100-1000 samples.
