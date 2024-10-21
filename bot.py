import base64
import pandas as pd
import yfinance as yf
import joblib
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import random
import time
from openai import OpenAI
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define functions for data fetching and prediction
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    latest_data = data.iloc[-1]
    return data, latest_data

# Define a simple linear regression model for price prediction
def predict_price(ticker, start_date, end_date, steps=1):
    ticker = ticker.upper()
    data, latest_data = fetch_data(ticker, start_date, end_date)
    historical_prices = data[['Adj Close']]
    historical_prices['Days'] = np.arange(len(historical_prices))

    # Train linear regression model
    X = historical_prices[['Days']]
    y = historical_prices['Adj Close']
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_days = np.arange(len(historical_prices), len(historical_prices) + steps).reshape(-1, 1)
    predicted_prices = model.predict(future_days)

    # Generate date range for predicted prices
    last_date = historical_prices.index[-1]
    future_dates = pd.date_range(start=last_date, periods=steps+1)[1:]
    predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Close'])

    return predicted_df, historical_prices

# Define the SMA strategy function
def sma_strategy(ticker, short_window, long_window):
    data = yf.download(ticker, period="1d", interval="1d")
    data['SMA_Short'] = data['Adj Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Adj Close'].rolling(window=long_window).mean()
    last_short_sma = data['SMA_Short'].iloc[-1]
    last_long_sma = data['SMA_Long'].iloc[-1]

    if last_short_sma > last_long_sma:
        return 'Buy'
    elif last_short_sma < last_long_sma:
        return 'Sell'
    else:
        return 'Hold'

# Streamlit UI with user inputs
def main():
    # Add a link to the style.css file
    with open("style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("Crypto Price Prediction and Analysis App")

    ticker = st.selectbox("Select a ticker symbol:", ["BTC-USD", "ETH-USD", "LTC-USD"])

    date_range = st.date_input("Select a date range:", value=(pd.Timestamp('2022-01-01'), pd.Timestamp.today()))

    short_window = st.slider("Short SMA Window:", min_value=1, max_value=100, value=10)
    long_window = st.slider("Long SMA Window:", min_value=1, max_value=200, value=50)

    prediction_steps = st.number_input("Prediction Steps (Linear Model):", value=1, min_value=1)

    show_current_price = st.checkbox("Show Current Price")
    show_predicted_price = st.checkbox("Show Predicted Price")
    show_sma_analysis = st.checkbox("Show SMA Analysis")

    if st.button("Predict"):
        if ticker:
            # Pass start_date and end_date to predict_price
            predicted_closing_price, historical_prices = predict_price(ticker, date_range[0], date_range[1], steps=prediction_steps)

            # Plot historical and predicted prices
            fig = go.Figure()

            # Plot historical prices
            fig.add_trace(go.Scatter(x=historical_prices.index, y=historical_prices['Adj Close'], mode='lines', name='Historical Prices'))

            # Plot predicted prices
            fig.add_trace(go.Scatter(x=predicted_closing_price.index, y=predicted_closing_price['Predicted Close'], mode='lines', name='Predicted Prices', line=dict(dash='dash')))

            # Update graph layout
            fig.update_layout(title=f'Historical and Predicted Prices for {ticker}',
                              xaxis_title='Date',
                              yaxis_title='Price',
                              legend_title='Legend',
                              showlegend=True)

            st.plotly_chart(fig)

            if show_predicted_price:
                st.write(f"Predicted Closing Price for {ticker}:", predicted_closing_price)
            if show_sma_analysis:
                decision = sma_strategy(ticker, short_window, long_window)
                st.write(f"Trading Decision for {ticker}:", decision)

# Execute the Streamlit app
if __name__ == '__main__':
    main()
