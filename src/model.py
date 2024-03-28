import os
import joblib
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def cross_val_model(data, model):
    x, y = get_data(data)    
    score = cross_val_score(model, x, y, cv=5, scoring='accuracy').mean()
    print(f'Cross validation score: {score}')

def get_load_model_path(model):
    model_path = f'./saved_models/'

    if model == 'logistic_regression':
        model_path = os.path.join(model_path, 'logistic_regression.joblib')
    elif model == 'random_forest':
        model_path = os.path.join(model_path, 'random_forest.joblib')
    elif model == 'ann':
        model_path = os.path.join(model_path, 'ann.joblib')
    else:
        raise Exception("Model type not implemented")

    return model_path


def get_model_name(model):
    if type(model) is LogisticRegression:
        return 'logistic_regression'
    elif type(model) is RandomForestClassifier:
        return 'random_forest'
    elif type(model) is MLPClassifier:
        return 'ann'
    else:
        raise Exception("Model type not implemented")

def compute_metrics(model, x, y):
    preds = model.predict(x)
    pred_probs = model.predict_proba(x)[:,1]
    
    # Get accuracy, precision, recall, f1, and roc_auc scores
    accuracy = model.score(x, y)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, pred_probs)
    fpr, tpr, thresholds = roc_curve(y, pred_probs)
    
    # Get displayed metrics 
    model_name = get_model_name(model)
    cm = ConfusionMatrixDisplay.from_predictions(y, preds)
    curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    metrics_path = f'./metrics/{model_name}'

    # Save CM
    cm.plot()
    plt.savefig(f"{metrics_path}/cm.jpg")
    plt.close()

    # Save ROC Curve
    curve.plot()
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.text(0.5, 0.5, f'Best Threshold: {best_threshold:.4f}', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.savefig(f"{metrics_path}/roc_curve.jpg")
    plt.close()

    # Save metrics
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'roc_auc': [roc_auc],
        'best_threshold': [best_threshold]
    })
    metrics_df.to_csv(f"{metrics_path}/metrics.csv", index=False, encoding='utf-8')

def get_saved_model_path(model):
    model_path = f'./saved_models/'

    if type(model) is LogisticRegression:
        model_path = os.path.join(model_path, 'logistic_regression.joblib')
    elif type(model) is RandomForestClassifier:
        model_path = os.path.join(model_path, 'random_forest.joblib')
    elif type(model) is MLPClassifier:
        model_path = os.path.join(model_path, 'ann.joblib')
    else:
        raise Exception("Model type not implemented")

    return model_path

def save_model(model):
    model_path = get_saved_model_path(model)
    joblib.dump(model, model_path)

def load_model(model):
    model_path = get_load_model_path(model)
    model= joblib.load(model_path)
    return model

def create_model(model):
    if model == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model == "random_forest":
        return RandomForestClassifier(max_depth=2, random_state=42)
    elif model == "ann":
        return MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42, verbose=True)

def get_feature_importance(model, x):
    feature_importance = model.coef_[0]
    feature_names = x.columns.tolist()

    feature_importance = list(zip(feature_names, feature_importance))
    feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

    feature_df = pd.DataFrame(feature_importance, columns=['feature', 'coef'])
    feature_df.to_csv('./data/feature_importance.csv', index=False, encoding='utf-8')

def train_model(model, x, y):
    model.fit(x, y)

    if type(model) is LogisticRegression:
        get_feature_importance(model, x)

    save_model(model)

def select_model(test, model_type):
    if test:
        return load_model(model_type)
    else:
        return create_model(model_type)

def get_data(data):
    if data is None or len(data) < 1:
        raise Exception("No data to train the model")
    elif "winner" not in data.columns:
        raise Exception("No target variable in the data")
    x = data.iloc[:, :-1]
    y = data['winner']
    return x, y

def run_testing(data, model):
    x, y = get_data(data)
    compute_metrics(model, x, y)

def run_training(data, model):   
    x, y = get_data(data)
    train_model(model, x, y)