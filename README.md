# PowerCo Churn Prediction

This project is a fictional case study that aims to predict which PowerCo customers are more likely to churn.

## Project Structure

- `scripts/`:
  - `preprocessing.py`: Script for preprocessing the training and testing data.
  - `train_dnn.py`: Script for training and evaluating a DNN.
  - `train_xgb.py`: Script for training and evaluating a XGBClassifier.
  - `predict.py`: Script for making predictions on a new set of data.

## Requirements

The following libraries are required:
- `xgboost`
- `scikit-learn`
- `tensorflow`

they can be installed with:
```sh
pip install xgboost scikit-learn tensorflow
```

## Usage

Preprocess the training and testing data by running:

```sh
python3 scripts/preprocessing.py
```
This script will generate `train_data.csv` and `test_data.csv` in the `data/processed/` directory. The input `csv` files are assumed to be stored in the `data/raw/` directory.

## Train models

Train the Deep Neural Network (DNN) model:

```sh
python3 scripts/train_dnn.py
```
This script will train the DNN model and save it as a `h5` file. Alternatively, a XGBoost (XGB) model can be trained with:

```sh
python3 scripts/train_xgb.py
```
This script will train the XGB model and save it as a `model` file. In both cases, the relevant plots are saved in the `plots/` directory.

## Making Predictions

Make predictions on a new dataset with:

```sh
python3 scripts/predict.py model_filename threshold model_type
```

Replace `model_filename` with the path to the trained model (e.g., `dnn_model.h5` or `xgboost_model.pkl`), `threshold` with the decision threshold (e.g., 0.5), and `model_type` with the type of model (dnn or xgboost). The predictions will be saved in `data/predictions.csv`.
