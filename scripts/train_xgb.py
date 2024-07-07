import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, precision_recall_curve, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

def load_and_train(input_data):

    train_data = pd.read_csv(input_data)

    X = train_data.drop(columns=['id', 'churn'])
    y = train_data['churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #balance class weights
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    #save feature names (for plots)
    feature_names = X.columns.tolist()

    #define the XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=1000, #high to use early stopping
        max_depth=5,
        min_child_weight=1,
        gamma=0.1, #minimum loss reduction for new partition on leaf node
        subsample=0.8, #fraction of samples (prevents overtraining)
        colsample_bytree=0.8, #fraction of features for fitting (prevents overtraining)
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1, #L2 regularization
        reg_alpha=0, #no L1 regularization
        learning_rate=0.01,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    #train
    eval_set = [(X_train, y_train), (X_val, y_val)]
    xgb_model.fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set, early_stopping_rounds=50, verbose=True)

    return xgb_model, feature_names, X_val, y_val

def evaluate(xgb_model, feature_names, X_val, y_val, optimal_threshold=0.5):

    y_val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
    y_val_pred = xgb_model.predict(X_val)
    roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    brier_score = brier_score_loss(y_val, y_val_pred_prob)

    print(f"XGBoost - ROC AUC: {roc_auc:.4f}, Brier Score: {brier_score:.4f}")
    print(classification_report(y_val, y_val_pred))

    #now with optimal threshold (from precision-recall curve)
    y_val_pred_adjusted = (y_val_pred_prob >= optimal_threshold).astype(int)

    print(f"Classification report for threshold {optimal_threshold}:")
    print(classification_report(y_val, y_val_pred_adjusted))

    model_filename = "xgboost_model.model"  
    xgb_model.get_booster().save_model(model_filename)

    return y_val_pred_prob, roc_auc

def main():

    input_data = 'data/processed/train_data.csv'
    model, feature_names, X_val, y_val = load_and_train(input_data)
    y_val_pred_prob, roc_auc = evaluate(model, feature_names, X_val, y_val, optimal_threshold=0.57)

    #plot ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve_xgb.png', bbox_inches='tight')
    plt.close()

    #plot precision-recall curve to optimize threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Precision-Recall vs Threshold")
    plt.savefig('plots/precision_recall_vs_threshold_xgb.png', bbox_inches='tight')
    plt.close()

    #plot top 10 features
    model.get_booster().feature_names = feature_names

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/top_10_feature_importance_xgb.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
