# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
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


# Step 1: Fetch the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
# Concatenate both training and test data
data = pd.concat([train_df,test_df]).reset_index(drop=True)

# FEATURE ENGINEERING
data = preprocess_data(data)
data = drop_columns(data)

# Split the training and test data again
train_df = data[data['Survived'].notnull()].reset_index(drop=True)
test_df = data[data['Survived'].isnull()].reset_index(drop=True)

# Step 2: # Define features (X) and target (y)
X = data.drop(columns=['Survived'])  # 'Survived' is the target
y = data['Survived']  # Target variable

# Split into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Step 4: Scale the features for better neural network performance
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

####################
# BUILDING A MODEL #
####################

# Step 5: Build the model function and compile it
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

# Step 6: Define Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 7: Train the model
model = build_model()
history = model.fit(X_train, y_train, 
    validation_data=(X_valid, y_valid),
    epochs=50, batch_size=10, 
    callbacks=[early_stop], 
    verbose=1
)

# Step 8: Evaluate model on the test data
test_loss, test_acc = model.evaluate(X_valid, y_valid, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")

# Generate predictions
# Model evaluation on the test set
y_pred = model.predict(X_valid)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class (0 or 1)

# Calculate Metrics
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_pred)
conf_matrix = confusion_matrix(y_valid, y_pred)

# Print Metrics
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)




 