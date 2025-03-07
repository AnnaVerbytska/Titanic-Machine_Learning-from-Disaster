import re
import traceback
import string
import numpy as np
import pandas as pd


#################################
# Feature Engineering Functions #
#################################


def fill_missing_values(data):
    """
    Fills missing values in the given DataFrame based on predefined rules:
    - 'Age' is filled with its median.
    - 'Fare' is filled with its mean and capped at 300.
    - 'Embarked' is filled with the most common value.
    - A new binary column 'Embarked_S' is created (1 if 'Embarked' is 'S', else 0).
    - 'Parch' is filled with the most common parent-child value.
    """
    data['Age'] = data['Age'].fillna(data['Age'].median(skipna=True))
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean(skipna=True)).clip(upper=300)
    data['Embarked'].fillna(data['Embarked'].value_counts().idxmax(), inplace=True)
    data['Embarked_S'] = np.where(data['Embarked'] == 'S', 1, 0)
    data['Parch'].fillna(data['Parch'].value_counts().idxmax(), inplace=True)
    
    return data

def calculate_family_size(data):
    """
    'FamilySize' is created by summing 'SibSp' and 'Parch' and adding 1 to account for the individual.
    """
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data

def convert_categorical_to_numerical(data):
    """
    Converts categorical columns to numerical columns using one-hot encoding.
    Uses drop_first to avoid collinearity
    """
    data = pd.get_dummies(data, columns=['Sex','Pclass'], dtype=int, drop_first=True)
    return data

def drop_columns(data):
    """
    Perform feature engineering by dropping non-interested columns.

    Parameters:
    data (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame after dropping specified columns.
    """
    data.drop(['Name', 'Cabin', 'Ticket', 'Embarked'], axis=1, inplace=True)
    return data

###################
# Master Function #
###################

def preprocess_data(data, fill_missing=True, family_size=True, convert_categorical=True):
    """
    Allows selecting specific preprocessing steps for experimentation.
    """
    if fill_missing:
        data = fill_missing_values(data)
    if family_size:
        data = calculate_family_size(data)
    if convert_categorical:
        data = convert_categorical_to_numerical(data)
    
    return data