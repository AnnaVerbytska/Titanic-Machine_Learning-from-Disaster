# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # scaling data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow import keras
from tensorflow.python.keras.models import Sequential # creating & training a model
from tensorflow.python.keras.layers import Dense, Dropout #Input
from tensorflow.python.keras.callbacks import EarlyStopping
# Cross-Validation
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow.python.keras.activations import sigmoid
import tensorflow as tf; tf.keras
# Settings
import sys
sys.path.append('../')
# Enable autoreload only in Jupyter
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
except (ImportError, AttributeError):
    pass  # Ignore if not in Jupyter

# Import feature engineering functions
from src.preprocessing import drop_columns


# Step 1: Load the concatenated cleaned data file (CSV)
data = pd.read_csv('cleaned_data.csv')

# Step 2: Separate features (X) and target (y)
X = data.drop(columns=['Survived'])  # Assuming 'Survived' is the target
y = data['Survived']  # Target variable

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 4: Scale the features for better neural network performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

####################
# BUILDING A MODEL #
####################

# Step 5: Build the model
def build_model():
    model=Sequential() #help(Sequential)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) # activation for binary classification problem

    #model.compile(optimizer='rmsprop', loss='binary_crossentropy') # loss for binary classification problem
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = build_model()

# Step 6: Define Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

####################
# CROSS-VALIDATION #
####################

# Stratified K-Fold Cross-Validation
estimate = KerasClassifier(build_fn=build_model, epochs=20, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimate, X, y, cv=kfold)
# Print the results
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Step 7: Train the model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=50, batch_size=10, 
    callbacks=[early_stop], 
    verbose=1
)

# Step 8: Evaluate model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")

# Generate predictions
y_pred_prob = model.predict(X_test)  # Get probability scores
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")





 