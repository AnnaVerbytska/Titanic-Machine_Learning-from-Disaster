# Import libraries for data analysis 
import pandas as pd
import numpy as np

# Import libraries for Logistic Regression model with L2 (Ridge) regularization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler # scaling is important when using regularization

# Fetch the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Fill in missing value Age with average
train_df['Age']=train_df['Age'].fillna(train_df['Age'].median()).infer_objects(copy=False)
# Delete Cabin column
train_df.drop('Cabin', axis=1, inplace=True)
# Fill missing values with the most frequent value (mode)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Remove outliers: Male survivors over 65
train_df = train_df.drop(train_df[(train_df['Sex'] == 'male') & 
                                    (train_df['Survived'] == 1) & 
                                    (train_df['Age'] > 63)].index)

# Convert 'Sex' and 'Embarked' to dummy variables (one-hot encoding)
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# Check new col names after one-hot encoding
print("Columns after one-hot encoding:")
print(train_df.columns)

# Define features (X) and target (y)
X = train_df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = train_df['Survived']

# Scale the features (scaling is important when using regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model with L2 regularization (Ridge)
logreg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')  # C is the inverse of regularization strength (smaller C = stronger regularization)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = logreg.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
