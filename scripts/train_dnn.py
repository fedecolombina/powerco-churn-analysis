import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, precision_recall_curve, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import pickle

def load_and_train(input_data):

    train_data = pd.read_csv(input_data)

    X = train_data.drop(columns=['id', 'churn']) #remove customer ID and output from training inputs
    y = train_data['churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #balance class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    #define DNN architecture
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])

    learning_rate = 0.0001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    #train
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                        class_weight=class_weights, callbacks=[early_stopping])

    return model, history, X_val, y_val

def evaluate(model, X_val, y_val):

    y_val_pred_prob = model.predict(X_val).ravel()
    y_val_pred = (y_val_pred_prob > 0.5).astype(int)
    roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    brier_score = brier_score_loss(y_val, y_val_pred_prob)

    print(f"DNN - ROC AUC: {roc_auc:.4f}, Brier Score: {brier_score:.4f}")
    print(classification_report(y_val, y_val_pred))

    #save the model
    model.save('dnn_model.h5')

    return y_val_pred_prob, y_val_pred, roc_auc

def main():

    input_data = 'data/processed/train_data.csv'
    model, history, X_val, y_val = load_and_train(input_data)
    y_val_pred_prob, y_val_pred, roc_auc = evaluate(model, X_val, y_val)

    #plot ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('plots/dnn_roc_curve.png')
    plt.close()

    #plot training and validation Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('plots/dnn_loss_curve.png')
    plt.close()

if __name__ == "__main__":
    main()
