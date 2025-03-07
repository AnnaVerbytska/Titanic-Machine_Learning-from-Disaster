# Import necessary libraries
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # scaling data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow import keras
from tensorflow.python.keras.models import Sequential # creating & training a model
from tensorflow.python.keras.layers import Dense #Input
from tensorflow.python.keras.callbacks import EarlyStopping
# Cross-Validation
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow.python.keras.activations import sigmoid
import tensorflow as tf; tf.keras
import warnings
warnings.filterwarnings("ignore")
# Settings
import sys
sys.path.append('../src')
# Import feature engineering functions
from preprocessing import preprocess_data, drop_columns

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Fix the 'DistributedDatasetInterface' Error
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

# Fetch the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Concatenate both training and test data
data = pd.concat([train_df,test_df]).reset_index(drop=True)

#######################
# FEATURE ENGINEERING #
#######################

# Preprocess the data
data = preprocess_data(data)
data = drop_columns(data)

# Split the training and test data again
train_df = data[data['Survived'].notnull()].reset_index(drop=True)
test_df = data[data['Survived'].isnull()].reset_index(drop=True)

# Drop PassengerId only from train_df (but keep it in test_df)
train_df = train_df.drop(columns=['PassengerId'], errors='ignore')

# Define features (X) and target (y)
X = train_df.drop(['Survived'], axis=1).values
y = train_df['Survived'].values

# Keep test_df features for final prediction
#X_test = test_df.drop(['Survived', 'PassengerId'], axis=1, errors='ignore').values  # Ensure no errors

# Split into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=SEED, stratify=y)

# Print the shape of the training and validation sets
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")

# Scale the features for better neural network performance
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

# Prepare test data: Drop 'Survived' (which is NaN in test_df) and apply the same scaling
X_test = scaler.transform(test_df.drop(columns=['Survived', 'PassengerId']).values)

####################
# BUILDING A MODEL #
####################

# Build the model function and compile it
def build_model():
    model=Sequential() #help(Sequential)
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) # activation for binary classification problem

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = KerasClassifier(model=build_model, epochs=20, batch_size=5, verbose=1)

# Define Early Stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

####################
# TRAINING A MODEL #
####################

# Train the model
model = build_model()
history = model.fit(X_train, y_train, 
    validation_data=(X_valid, y_valid),
    epochs=50, batch_size=10, 
    callbacks=[early_stop], 
    verbose=1,
    shuffle=False
)

##############
# EVALUATION #
##############

# Model evaluation on the test set
y_pred = model.predict(X_valid)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class (0 or 1)

###########
# METRICS #
###########

# Calculate the metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_pred)
conf_matrix = confusion_matrix(y_valid, y_pred)

# Print the metrics
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

#####################
# KAGGLE SUBMISSION #
#####################

# Predict on Test Set
test_predictions = (model.predict(X_test) > 0.5).astype(int)

# Create Submission File
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions.flatten()
})

submission.to_csv('../data/submission_keras.csv', index=False)

print("\n✅ Submission file 'submission_keras.csv' saved successfully!")




 