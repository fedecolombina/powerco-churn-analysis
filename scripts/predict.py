import pandas as pd
import numpy as np
import pickle
import sys
from tensorflow.keras.models import load_model
import xgboost as xgb

def load_dnn_model(model_filename):
    return load_model(model_filename)

def load_xgb_model(model_filename):
    try:
        model = xgb.Booster()  #initialize XGBoost model
        model.load_model(model_filename)
        return model
    except xgb.core.XGBoostError as e:
        print(f"Error loading XGBoost model: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/predict.py <model_filename> <threshold> <model_type>")
        sys.exit(1)

    model_filename = sys.argv[1]
    threshold = float(sys.argv[2])  #optimized threshold
    model_type = sys.argv[3]  #'dnn' or 'xgboost'

    test_data = pd.read_csv('data/processed/test_data.csv')

    if 'id' not in test_data.columns:
        print("Error: 'id' column not found in test data.")
        sys.exit(1)

    X_test = test_data.drop(columns=['id'])

    #predict
    if model_type == 'dnn':
        model = load_dnn_model(model_filename)
        test_pred_prob = model.predict(X_test).ravel()
    elif model_type == 'xgboost':
        model = load_xgb_model(model_filename)
        dtest = xgb.DMatrix(X_test)  #use DMatrix to handle input data
        test_pred_prob = model.predict(dtest)
    else:
        raise ValueError("Unknown model type. Choose between 'dnn' or 'xgboost'.")

    #apply threshold to get predictions
    test_pred = (test_pred_prob > threshold).astype(int)

    #save
    results = pd.DataFrame({
        'id': test_data['id'],
        'churn_prob': test_pred_prob,
        'churn': test_pred
    })

    results.sort_values(by='churn_prob', ascending=False, inplace=True)
    results.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to 'data/predictions.csv'")

if __name__ == "__main__":
    main()
