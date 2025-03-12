import re
import traceback
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


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


###########################
# VISUALISATION FUNCTIONS #
###########################

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#def eda-visualisation(train, test, age_distribution=True, age_gender_distribution=True):

# Visualise age distribution in train and test sets
def age_distribution(train, test):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Data", "Test Data"))
    fig.add_trace(
        go.Histogram(x=train["Age"], nbinsx=50, name="Train", marker=dict(color='blue'), opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=test["Age"], nbinsx=50, name="Test", marker=dict(color='red'), opacity=0.7),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Age Distribution in Train and Test Datasets",
        showlegend=False,  # Set to True if you want legends
        bargap=0.1,
    )
    return fig

# Visualise the age distribution of the survivors across gender in a train set
def age_gender_distribution(train):
    fig = px.scatter(train, x='Age', y='Survived', color='Sex', 
                    color_discrete_map={'male': 'light blue', 'female': 'orange'}, 
                    symbol='Sex', labels={'Sex': 'Gender'})
    fig.update_layout(
        xaxis_title='Age & Survived by Gender',
        yaxis_title='Survival (0 = No, 1 = Yes)',
        showlegend=True,
        legend_title='Gender'
    )
    
    return fig

# Visualise fare distribution in train and test sets
def fare_distribution(train, test):

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Data", "Test Data"))
    fig.add_trace(
        go.Histogram(x=train["Fare"], nbinsx=50, name="Train", marker=dict(color='blue'), opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=test["Fare"], nbinsx=50, name="Test", marker=dict(color='red'), opacity=0.7),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Fare Distribution in Train and Test Datasets",
        showlegend=False,  # Set to True if you want legends
        bargap=0.1,
    )
    return fig

# Visualise the survivors across age, fare, class, and sex features
def survivors_3d_plot(train):
    fig = px.scatter_3d(
            train,
            x='Age',  
            y='Fare',  
            z='Pclass',  
            color="Survived",
            title="Survivors - 3D Plot",
            labels={"Survived": "Survival (0 = No, 1 = Yes)"},
            color_continuous_scale=["gray", "green"],
            hover_data=train.columns,
             size_max=10,
            opacity=0.8
            )
    fig.update_layout(
            title="Age & Fare & Class & Gender across Survivors",
            title_x=0.5, 
            width=1000,
            height=600,
            showlegend=True
            )
    return fig


# Visualise the relationship among age, Pclass, and survivors features
def age_pclass_survivors(train):
    df_corr= train[['Survived', 'Age', 'Pclass']]
    corr = df_corr.corr()
    fig = px.imshow(corr, text_auto=True)
    fig.update_layout(
        title='Correlation Heatmap of Age & Pclass & Survivors',
        width=770,
        height=500  
    )
    return fig

# Visualise the relationship between the number of siblings/spouses and parents/children in a train set
def sibsp_parch_heatmap(train):
    pivot_table = train.pivot_table(values='Survived', index='Parch', columns='SibSp', aggfunc='mean')
    # Plot the heatmap
    fig = px.imshow(pivot_table, 
                labels={'x': 'Number of Siblings/Spouses', 'y': 'Number of Parents/Children', 'color': 'Survival Rate'}, 
                title='Survival Rate Heatmap by SibSp & Parch',
                text_auto=True)
    fig.update_layout(
    width=770,
    height=500  
    )
    # Add a spiral line plot
    theta = np.linspace(0, 4 * np.pi, 100)
    r = 5 + theta  # Radius grows with theta to create a spiral
    x_spiral = r * np.cos(theta)  # x = r * cos(θ)
    y_spiral = r * np.sin(theta)  # y = r * sin(θ)
    # Adjust the x and y limits to fit the heatmap
    x_spiral = np.interp(x_spiral, (x_spiral.min(), x_spiral.max()), (0, pivot_table.shape[1] - 1))
    y_spiral = np.interp(y_spiral, (y_spiral.min(), y_spiral.max()), (0, pivot_table.shape[0] - 1))
    # Add the spiral plot to the figure
    fig.add_trace(go.Scatter(x=x_spiral, y=y_spiral, mode='lines', line=dict(color='white', width=3), name='Spiral Line'))
    return fig


# Function: Convert Matplotlib Figure to Image URI
def fig_to_uri(fig, dpi=2000):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# Visualise the relationaship between age and fare in a test set 
def age_fare_jointplot(test, dpi=2000):
    g = sns.JointGrid(data=test, x='Age', y='Fare', height=8)
    g.plot_joint(sns.kdeplot, cmap="Greens", fill=True)
    g.plot_marginals(sns.kdeplot, color='green', fill=True)
    g.ax_joint.set_title('Age & Fare', fontsize=10, pad = 80)
    fig=g.figure
    return fig_to_uri(fig, dpi=dpi)
