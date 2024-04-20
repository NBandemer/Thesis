import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import model as m
import util as u

scaler = StandardScaler()

new_cols = ['w_first_serve_pt', 'w_first_serve_won', 'w_second_serve_won', 'w_double_faults', 'w_aces', 'w_break_points_saved', 'w_break_points_faced', 'w_return_first_serve_pt_won', 'w_return_second_serve_won', 'w_bp_converted', 'w_bp_opportunities', 'l_first_serve_pt', 'l_first_serve_won', 'l_second_serve_won', 'l_aces', 'l_double_faults', 'l_break_points_saved', 'l_break_points_faced', 'l_return_first_serve_pt_won', 'l_return_second_serve_won', 'l_bp_converted', 'l_bp_opportunities', 'w_h2h', 'l_h2h', 'w_match_difficulty', 'l_match_difficulty']
old_cols = ['tourney_id', 'tourney_date', 'match_num', 'surface', 'draw_size', 'tourney_level', 'winner_hand', 'loser_hand', 'best_of', 'round', 'winner_age', 'loser_age',  'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points', 'winner_id', 'loser_id']
features = new_cols + old_cols
df = pd.read_csv('data/atp_matches_1991-2023_with_refined_stats.csv', usecols=features)
df = df.dropna(subset=features)
df = u.anonymize_data(df)

years = df['tourney_id'].str[:4].astype(int)
df['year'] = years

df_train, df_test = u.test_train_split_by_year(df)
df_train, df_val = u.test_train_split_by_year(df_train)

df_train.drop(columns=['tourney_id', 'tourney_date', 'match_num', 'year'], inplace=True)
df_val.drop(columns=['tourney_id', 'tourney_date', 'match_num',  'year'], inplace=True)
df_test.drop(columns=['tourney_id', 'tourney_date', 'match_num',  'year'], inplace=True)

X_train, y_train = m.get_data(df_train, model=None)
X_val, y_val = m.get_data(df_val, model=None)
X_test, y_test = m.get_data(df_test, model='xgb')

numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cats = X_train.select_dtypes(exclude=np.number).columns.tolist()

train_categorical = X_train[cats]
val_categorical = X_val[cats]
test_categorical = X_test[cats]

# Convert to Pandas category
for col in cats:
    train_categorical[col] = train_categorical[col].astype('category')
    val_categorical[col] = val_categorical[col].astype('category')
    test_categorical[col] = test_categorical[col].astype('category')


x_scaled = scaler.fit_transform(X_train[numerical_cols])
x_val_scaled = scaler.transform(X_val[numerical_cols])
x_test_scaled = scaler.transform(X_test[numerical_cols])

numerical_train_df = pd.DataFrame(x_scaled, columns=numerical_cols)
numerical_val_df = pd.DataFrame(x_val_scaled, columns=numerical_cols)
numerical_test_df = pd.DataFrame(x_test_scaled, columns=numerical_cols)

# Reset indices
train_categorical.reset_index(drop=True, inplace=True)
val_categorical.reset_index(drop=True, inplace=True)
test_categorical.reset_index(drop=True, inplace=True)

numerical_train_df.reset_index(drop=True, inplace=True)
numerical_val_df.reset_index(drop=True, inplace=True)
numerical_test_df.reset_index(drop=True, inplace=True)

X_train =  pd.concat([train_categorical, numerical_train_df], axis=1)
X_val =  pd.concat([val_categorical, numerical_val_df], axis=1)
X_test =  pd.concat([test_categorical, numerical_test_df], axis=1)

import xgboost as xgb

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Define hyperparameters
params = {"objective": "binary:logistic", "device": "cuda", "eta": "0.05", "eval_metric": "logloss", "max_depth": "6", "subsample": "1", "random_state": "42"}

param_grid = {
    "objective": ["binary:logistic"],
    "eta": [0.05, 0.1, 0.3],
    "max_depth": [3, 6, 9],
    "subsample": [0.5, 0.75, 1],
    "random_state": [42]
}

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier(device="cuda", eval_metric="logloss")

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring="neg_log_loss", cv=5, n_jobs=-1)
grid_search.fit(dtrain_reg, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found:", best_params)

n = 10000
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   early_stopping_rounds=10,
    evals=[(dval, "validation")],  # Specify the validation set
    verbose_eval=True,
)

# Predict probabilities for the test set
y_pred_prob = model.predict(dtest_reg)

# Convert probabilities to binary predictions
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute metrics
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Get displayed metrics 
model_name = 'xgb'
cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
best_threshold = thresholds[np.argmax(tpr - fpr)]

metrics_path = f'./metrics/{model_name}'
metrics_path = os.path.join(metrics_path, '')
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
