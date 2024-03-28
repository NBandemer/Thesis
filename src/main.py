import pandas as pd
import model as m
import util as u

test, cv, model = u.get_config()

if test and cv:
    print('Cannot run test and cv at the same time!')
    exit(1)

model = m.select_model(test, model)
train_data = None
test_data = None

# Load test or training data depending on mode
if test:
    test_data = pd.read_csv('./data/test_train_split/test.csv').dropna()
else:
    train_data = pd.read_csv('./data/test_train_split/train.csv').dropna()

# Run cross validation, testing, or training depending on mode
if cv and train_data is not None:
    m.cross_val_model(data=train_data, model=model)
elif test and test_data is not None:
    m.run_testing(data=test_data, model=model)
elif not test and train_data is not None:
    m.run_training(data=train_data, model=model)
else:
    print('Invalid mode or no data to train/test the model!')

