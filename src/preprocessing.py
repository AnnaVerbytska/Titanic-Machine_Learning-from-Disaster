import re
import traceback
import string
import numpy as np
import pandas as pd

###################
# Master Function #
###################



################################
# DataFrame cleaning functions #
################################

def drop_columns(data):
    """
    Perform feature engineering by dropping non-interested columns.

    Parameters:
    data (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame after dropping specified columns.
    """
    data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
    return data