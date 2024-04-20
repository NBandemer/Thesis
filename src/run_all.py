import pandas as pd
from sklearn.linear_model import LogisticRegression
import util as u
import model as m

test, cv, _, pca = u.get_config()
models = ['logr', 'lgbm']

if test and cv:
    print('Cannot run test and cv at the same time!')
    exit(1)

for model_name in models:
    # Create the model
    model = m.select_model(False, model_name)

    # Load train / test data, depending on the model
    data_path = './data/rank_points' if type(model) == LogisticRegression else './data/rank'
    test_data = pd.read_csv(f'{data_path}/test.csv').dropna()
    train_data = pd.read_csv(f'{data_path}/train.csv').dropna()
    
    # TRAINING
    m.run_training(data=train_data, model=model, pca=pca)

    # TESTING
    model = m.select_model(True, model_name)
    m.run_testing(data=test_data, train_data=train_data, model=model, pca=pca)

    # CROSS VALIDATION
    m.cross_val_model(data=train_data, model_name=model_name, pca=pca)
    