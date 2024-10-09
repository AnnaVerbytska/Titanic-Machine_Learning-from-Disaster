# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential # creating & training a model
from tensorflow.python.keras.layers import Dense

# PREPARING THE DATA

# Fetch the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

X = train_df[['Pclass', 'Fare', 'SibSp', 'Parch']].values
y = train_df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# CREATING AND TRAINING A MODEL


 