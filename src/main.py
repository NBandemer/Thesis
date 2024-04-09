import pandas as pd
import model as m
import util as u

test, cv, model, time_split = u.get_config()

if test and cv:
    print('Cannot run test and cv at the same time!')
    exit(1)

model = m.select_model(test, model, time_split)
train_data = None
test_data = None

# Load test or training data depending on mode
data_path = './data/test_train_split' if not time_split else './data/time_split'

if test:
    test_data = pd.read_csv(f'{data_path}/test.csv').dropna()

train_data = pd.read_csv(f'{data_path}/train.csv').dropna()

# Run cross validation, testing, or training depending on mode
if cv and train_data is not None:
    m.cross_val_model(data=train_data, model=model)
elif test and test_data is not None:
    m.run_testing(data=test_data, train_data=train_data, model=model, time_split=time_split)
elif not test and train_data is not None:
    m.run_training(data=train_data, model=model, time_split=time_split)
else:
    print('Invalid mode or no data to train/test the model!')

