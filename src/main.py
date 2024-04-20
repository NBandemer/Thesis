import pandas as pd
from sklearn.linear_model import LogisticRegression
import model as m
import util as u

test, cv, model_name, pca = u.get_config()

if test and cv:
    print('Cannot run test and cv at the same time!')
    exit(1)

model = m.select_model(test, model_name)
train_data = None
test_data = None

# Load test or training data depending on mode
data_path = './data/rank_points' if type(model) == LogisticRegression else './data/rank'

if test:
    test_data = pd.read_csv(f'{data_path}/test.csv').dropna()

train_data = pd.read_csv(f'{data_path}/train.csv').dropna()

# Run cross validation, testing, or training depending on mode
if cv and train_data is not None:
    m.cross_val_model(data=train_data, model_name=model_name, pca=pca)
elif test and test_data is not None:
    m.run_testing(data=test_data, train_data=train_data, model=model, pca=pca)
elif not test and train_data is not None:
    print('Training model...')
    m.run_training(data=train_data, model=model, pca=pca)
    print('Model trained successfully!')
else:
    print('Invalid mode or no data to train/test the model!')

