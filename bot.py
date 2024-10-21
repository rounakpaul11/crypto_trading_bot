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

# Load the ARIMA model
arima_model = joblib.load('arima_model.joblib')

# Fetch the data from yfinance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the given ticker or date range.")
    latest_data = data.iloc[-1]
    return latest_data

# Predict future prices using ARIMA model
def predict_price(ticker, start_date, end_date, steps=1):
    ticker = ticker.upper()
    latest_data = fetch_data(ticker, start_date, end_date)
    
    if latest_data.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data

    adj_closing_price = latest_data['Adj Close']

    # Get forecast for the next steps
    forecast = arima_model.get_forecast(steps=steps)  
    predicted_residuals = forecast.predicted_mean
    predicted_close = adj_closing_price + np.cumsum(predicted_residuals.values)

    # Generate date range for the predicted prices
    date_range = pd.date_range(start=end_date, periods=steps + 1)
    predicted_prices = pd.DataFrame(predicted_close, index=date_range[1:], columns=['Predicted Close'])

    return predicted_prices

# Simple Moving Average strategy (SMA)
def sma_strategy(ticker, short_window, long_window):
    data = yf.download(ticker, period="1y", interval="1d")  # Fetch 1 year of data for SMA calculation
    if data.empty:
        return "No data"

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

# Response generator for chatbot
def response_generator():
    responses = [
        "Hello there! How can I assist you today?",
        "Hi, human! Is there anything I can help you with?",
        "Do you need help?",
    ]
    response = random.choice(responses)
    return response

# Streamlit UI with user inputs
def main():
    # Remove style.css if not using any styling
    # with open("style.css") as f:
    #     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("Crypto Price Prediction and Analysis App")

    ticker = st.selectbox("Select a ticker symbol:", ["BTC-USD", "ETH-USD", "LTC-USD"])

    date_range = st.date_input("Select a date range:", value=(pd.Timestamp('2022-01-01'), pd.Timestamp.today()))

    short_window = st.slider("Short SMA Window:", min_value=1, max_value=100, value=10)
    long_window = st.slider("Long SMA Window:", min_value=1, max_value=200, value=50)

    prediction_steps = st.number_input("Prediction Steps (ARIMA):", value=1, min_value=1)

    show_current_price = st.checkbox("Show Current Price")
    show_predicted_price = st.checkbox("Show Predicted Price")
    show_sma_analysis = st.checkbox("Show SMA Analysis")

    if st.button("Predict"):
        if ticker:
            # Fetch and predict
            predicted_closing_price = predict_price(ticker, date_range[0], date_range[1], steps=prediction_steps)

            if show_current_price:
                latest_data = fetch_data(ticker, start_date=date_range[0], end_date=date_range[1])
                current_price = latest_data['Adj Close']
                st.write(f"Current Price for {ticker}:", current_price)

            # Fetch historical data for plotting
            data = yf.download(ticker, start=date_range[0], end=date_range[1])

            # Plot historical and predicted prices
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Historical Prices'))

            if not predicted_closing_price.empty:
                fig.add_trace(go.Scatter(x=predicted_closing_price.index, y=predicted_closing_price['Predicted Close'], mode='lines', name='Predicted Prices'))

            fig.update_layout(title=f'Historical and Predicted Prices for {ticker}', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig)

            if show_predicted_price:
                st.write(f"Predicted Closing Price for {ticker}:", predicted_closing_price)

            if show_sma_analysis:
                decision = sma_strategy(ticker, short_window, long_window)
                st.write(f"Trading Decision for {ticker}:", decision)

# Execute the Streamlit app
if __name__ == '__main__':
    main()

# Chatbot integration as a sidebar
st.sidebar.title("Simple Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.sidebar:
        with st.empty():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Accept user input
if prompt := st.sidebar.text_input("Chat with me:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar:
        with st.empty():
            with st.chat_message("assistant"):
                response = response_generator()
                st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
