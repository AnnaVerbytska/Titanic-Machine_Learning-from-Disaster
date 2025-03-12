# Import libraries
import dash
from dash import dcc, html
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import io
import base64

import sys
sys.path.append('../src')
# Import feature engineering functions
from preprocessing import age_distribution, age_gender_distribution, age_pclass_survivors, fare_distribution, survivors_3d_plot, sibsp_parch_heatmap, age_fare_jointplot

# Fetch the data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Initialise Dash app
app = dash.Dash(__name__)

# Function to Convert Matplotlib Plot to Image
def fig_to_uri(fig):
    """ Convert Matplotlib figure to PNG image URI """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# App layout
app.layout = html.Div([
    html.H1("Titanic Exploratory Data Analysis Dashboard", style={'text-align': 'center'}),
    
    # Age Distribution
    html.H3("Age Distribution in Train and Test Datasets"),
    dcc.Graph(figure=age_distribution(train, test)),

    # Fare Distribution
    html.H3("Fare Distribution in Train and Test Datasets"),
    dcc.Graph(figure=fare_distribution(train, test)),

    # Age Fare Jointplot
    html.H3("Age & Fare Distribution in a Test Dataset"),
    html.Img(src=age_fare_jointplot(test), style={"width": "80%", "display": "block", "margin": "auto"}),

    # Age_gender_distribution
    html.H3("Age & Gender Distribution in a Train Dataset"),
    dcc.Graph(figure=age_gender_distribution(train)),

    # Age_pclass_survivors
    html.H3("Age & Pclass Distribution in a Train Dataset"),
    dcc.Graph(figure=age_pclass_survivors(train)),

    # Survivors 3D Plot
    html.H3("Age & Fare & Pclass & Gender Distribution in a Train Dataset"),
    dcc.Graph(figure=survivors_3d_plot(train)),
    
    # Sibsp Parch Heatmap
    html.H3("Siblings & Spouses and Parents & Children Distribution in a Train Dataset"),
    dcc.Graph(figure=sibsp_parch_heatmap(train))
    ])

# Run the App
if __name__ == '__main__':
    app.run_server(debug=True)