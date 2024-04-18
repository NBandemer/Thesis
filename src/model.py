import os
import joblib
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

# Keep this global so it can be later used to transform test data
scaler = StandardScaler()

supported_models = 'logr', 'lgbm', 'xgb'

columns = []

def cross_val_model(data, model):
    x, y = get_data(data)   
    x_scaled = scaler.fit_transform(x)
    accuracies = cross_val_score(model, x_scaled, y, cv=5, scoring='accuracy')
    accuracies_df = pd.DataFrame(accuracies, columns=['accuracy'])
    model_cv_path = f'./metrics/{get_model_name(model)}/'
    accuracies_df.to_csv(f'{model_cv_path}/cv_acc.csv', index=False, encoding='utf-8')

def get_load_model_path(model):
    file_path = f'./saved_models/refined/{model}.joblib'
    
    if model not in supported_models:
        raise Exception("Model type not implemented")
    elif os.path.exists(file_path):
        return file_path
    else:
        raise Exception("Model not found")

def get_model_name(model):
    if type(model) is LogisticRegression:
        return 'logr'
    elif type(model) is lgb.LGBMClassifier:
        return 'lgbm'
    elif type(model) is xgb.XGBClassifier:
        return 'xgb'
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
    metrics_path = os.path.join(metrics_path, 'refined/')
    os.makedirs(metrics_path, exist_ok=True)

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
    model_path = f'./saved_models/refined/'

    if type(model) is LogisticRegression:
        model_path = os.path.join(model_path, 'logr.joblib')
    elif type(model) is lgb.LGBMClassifier:
        model_path = os.path.join(model_path, 'lgbm.joblib')
    elif type(model) is xgb.XGBClassifier:
        model_path = os.path.join(model_path, 'xgb.joblib')
    else:
        raise Exception("Model type not implemented")

    return model_path

def save_model(model):
    model_path = get_saved_model_path(model)
    joblib.dump(model, model_path)

def load_model(model):
    model_path = get_load_model_path(model)
    model = joblib.load(model_path)
    return model

def create_model(model, seed=42):
    if model == "logr":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif model == "lgbm":
        return lgb.LGBMClassifier(random_state=seed)
    elif model == "xgb":
        return xgb.XGBClassifier(random_state=seed)

def get_feature_importance(model):
    if type(model) is LogisticRegression:
        feature_importance = model.coef_[0]
        path = './metrics/logr/refined/feature_importance.csv'
    elif type(model) is lgb.LGBMClassifier:
        feature_importance = model.feature_importances_
        path = './metrics/lgbm/refined/feature_importance.csv'

    feature_names = columns

    feature_importance = list(zip(feature_names, feature_importance))
    feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

    feature_df = pd.DataFrame(feature_importance, columns=['feature', 'coef'])
    feature_df.to_csv(path, index=False, encoding='utf-8')

def train_model(model, x, y):
    model.fit(x, y)

    if type(model) is LogisticRegression or type(model) is lgb.LGBMClassifier:
        get_feature_importance(model)

    save_model(model)

def select_model(test, model_type):
    if test:
        return load_model(model_type)
    else:
        return create_model(model_type)

def get_differentials(data):
    cols = ['first_serve_pt', 'first_serve_won', 'second_serve_won', 'double_faults', 'aces', 'break_points_saved', 'break_points_faced', 'return_first_serve_pt_won', 'return_second_serve_won', 'bp_converted', 'bp_opportunities', 'match_difficulty']
    for col in cols:
        data[col] = data['player1_' + col] - data['player0_' + col]
        data = data.drop(columns=['player0_' + col, 'player1_' + col])
    return data

def get_data(data):
    if data is None or len(data) < 1:
        raise Exception("No data to train the model")
    elif "winner" not in data.columns:
        raise Exception("No target variable in the data")

    # From ChatGPT to make sure number of instances where winner is 0 and 1 are equal
    min_count = data['winner'].value_counts().min()
    balanced_df = pd.concat([
        data[data['winner'] == 0].sample(n=min_count, random_state=1),
        data[data['winner'] == 1].sample(n=min_count, random_state=1)
    ])

    dropped = ['round', 'best_of', 'draw_size', 'day_sin', 'day_cos']
    levels = ['A','D','F','G','M']
    surfaces = ['Carpet', 'Clay', 'Hard', 'Grass']

    if get_model_name(model) == 'logr':
        for surface in surfaces:
            dropped.append('surface_' + surface)
        
        for level in levels:
            dropped.append('tourney_level_' + level)

        balanced_df = balanced_df.drop(columns=dropped)
    
    x = balanced_df.drop(columns=['winner'])
    y = balanced_df['winner']

    return x, y

def run_testing(data, train_data, model):
    x, y = get_data(data)

    try:
        means = scaler.mean_
    except AttributeError:
        X_train, _ = get_data(train_data)
        scaler.fit_transform(X_train)
    
    x_scaled = scaler.transform(x)
    compute_metrics(model, x_scaled, y)

def run_training(data, model):   
    global columns
    x, y = get_data(data)
    columns = x.columns
    x_scaled = scaler.fit_transform(x) if scaler is not None else exit(1)
    train_model(model, x_scaled, y)