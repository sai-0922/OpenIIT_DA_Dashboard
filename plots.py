import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import streamlit as st
df1=pd.read_csv('interpolated.csv',header=None)
df2=pd.read_csv('Final_whole_data.csv')
df3=pd.read_csv('y_original.csv')
df4=pd.read_csv('sarima_predictions.csv')
df5=pd.read_csv('reg_train_score_.csv')
df6=pd.read_csv('loss_history_.csv')

def pilplot():
    trace1 = go.Scatter(
        x=df1[0],  # X-axis can be the DataFrame index or another column if applicable
        y=df1[1],  # Y-axis corresponds to the selected column
        mode='lines',  # Display lines and markers
        name="pilgrims to tirupati",  # Legend label
    )
    # Create a layout for the plot (optional)
    layout = go.Layout(
        xaxis=dict(title='Dates'),
        yaxis=dict(title='Number of tourists'),
    )
    layout['title_font'] = dict(size=26)
    fig = go.Figure(data=[trace1], layout=layout)
    fig.update_layout(title_text='Actual Pilgrim Count to Tirupati',title_x=0.4, width=1400, height=800)

    return fig 


def featureplot():
    fig = make_subplots(rows=4, cols=3, subplot_titles=["Tirupati Rooms", "Tirupati Temple", "TTD", "Venkateswara yt", "Venkateswara swamy",
                                                        "Tirupati Darshan timings", "tirumala", "train to tiruapti", "tiruapti", "tirupati train",
                                                        "kanipakam", "tirupati distance"])

    # Define the data traces for each feature
    traces = []
    features = df2.columns[1:13]  # Assuming the first column is 'Date'

    for i, feature in enumerate(features):
        trace = go.Scatter(x=df2['Date'], y=df2[feature], mode='lines', name=feature)
        traces.append(trace)

    # Assign the traces to the subplots in the grid
    for i, trace in enumerate(traces):
        row = (i // 3) + 1
        col = (i % 3) + 1
        fig.add_trace(trace, row=row, col=col)

    # Update the layout
    fig.update_layout(width=1600, height=2000)
    return fig 



def heatmap():
    corr_matrix=df2.iloc[:,1:].corr()
    fig = px.imshow(corr_matrix,text_auto=True)
    fig.update_layout(width=1400, height=1200)
    return fig


def seasonality():
    result = seasonal_decompose(df2['Pilgrims'],model='additive',period=52)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                      subplot_titles=['Observed', 'Trend','Seasonal', 'Residual'])

    # Add the observed component to the first subplot
    fig.add_trace(go.Scatter(x=df2.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
    # Add the seasonal component to the second subplot
    fig.add_trace(go.Scatter(x=df2.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    # Add the trend component to the third subplot
    fig.add_trace(go.Scatter(x=df2.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
    # Add the residual component to the fourth subplot
    fig.add_trace(go.Scatter(x=df2.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)


    # Set the layout
    fig.update_layout(yaxis_title='Value',
                    width=1400, height=1200)

    return fig


def ACFPlot():
    # Create an ACF plot
    acf_values, acf_confint = acf(df1[1], nlags=100, alpha=0.05)

    # Create an interactive ACF plot using Plotly
    interactive_acf = go.Figure()

    interactive_acf.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'))

    interactive_acf.update_layout(
        xaxis_title="Lags",
        yaxis_title="ACF Value",
        width=1400,
        height=700,
    )
    # interactive_acf.show()
    # Show the interactive plot
    st.plotly_chart(interactive_acf)
    
    return

def plot_original_vs_predicted():
    # Create a Figure
    fig = go.Figure()

    # Add traces for original values and predicted values
    fig.add_trace(go.Scatter(x=df3['date'], y=df3['1'], mode='lines', name='Original Values'))
    fig.add_trace(go.Scatter(x=df3['date'], y=df3['0'], mode='lines', name='Final Predicted Values Ensemble',line=dict(color='red')))

    # Customize the layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Values",
        width=1400,
        height=700,
    )

    return fig

def sarima_nhits_kalman():
    fig = go.Figure()

    # Add traces for original values and predicted values
    fig.add_trace(go.Scatter(x=df4['date'], y=df4['sarima'], mode='lines', name='Sarima'))
    fig.add_trace(go.Scatter(x=df4['date'], y=df4['Nhits'], mode='lines', name='Nhits'))
    fig.add_trace(go.Scatter(x=df4['date'], y=df4['kalman'], mode='lines', name='kalman'))
    fig.add_trace(go.Scatter(x=df4['date'], y=df4['labels'], mode='lines', name='labels (Original Data)'))
    fig.add_trace(go.Scatter(x=df3['date'][-10:], y=df3['0'][-10:], mode='lines', name='ensemble'))

    # Customize the layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Values",
        width=1400,
        height=700,
    )

    return fig

def xgboost():
    boosting_iterations = np.arange(600) + 1
    training_deviance = df5['0']
    test_deviance = df5['1']

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for training and test deviance
    fig.add_trace(go.Scatter(x=boosting_iterations, y=training_deviance, mode='lines', name="Training Set Deviance", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=boosting_iterations, y=test_deviance, mode='lines', name="Test Set Deviance", line=dict(color='red')))

    # Customize the layout
    fig.update_layout(
        xaxis_title="Boosting Iterations",
        yaxis_title="Deviance",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        width=1400,
        height=700,
    )

    return fig

def accuracy():
    data = {'Model': ['Gradient Boosting Regressor', 'LSTM', 'Ensemble Model', 'Sarima', 'n-Hits'],
            'MAPE': [12.06, 4.9, 8.43, 5.3, 4.6]}
    df = pd.DataFrame(data)

    return df