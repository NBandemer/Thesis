from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import model as m
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
optimizer = Adam(learning_rate=0.00005)

train = pd.read_csv('./data/time_split/train.csv')
test = pd.read_csv('./data/time_split/test.csv')

# X_train, y_train = m.get_data(train)
# X_train = scaler.fit_transform(X_train) if scaler is not None else exit(1)

# X_test, y_test = m.get_data(test)
# X_test = scaler.transform(X_test)
X_train = train.loc[['player0_rank_points','player1_rank_points']]
y_train = train['winner']

X_test = test.loc[['player0_rank_points','player1_rank_points']]
y_test = test['winner']

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

num_features = X_train.shape[1]

# Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.45),  # Add dropout for regularization
    Dense(128, activation='relu'),
    Dropout(0.45),  # Add dropout for regularization
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=59, batch_size=32, validation_data=(X_val, y_val), callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
