import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import random

# Define functions for data fetching and prediction
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    latest_data = data.iloc[-1]
    return data, latest_data

# Simple Linear Regression model for price prediction
def predict_price(ticker, start_date, end_date, steps=1):
    ticker = ticker.upper()
    data, latest_data = fetch_data(ticker, start_date, end_date)

    # Prepare the data for linear regression
    data['Date'] = np.arange(len(data))  # Convert dates into integers for regression
    X = data[['Date']]
    y = data['Adj Close']
    
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_dates = np.arange(len(data), len(data) + steps).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)

    # Generate date range for the predicted prices
    date_range = pd.date_range(start=end_date, periods=steps+1)[1:]
    predicted_prices_df = pd.DataFrame(predicted_prices, index=date_range, columns=['Predicted Close'])

    return predicted_prices_df

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

# Chatbot response generator
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
    st.title("Crypto Price Prediction and Analysis App")

    ticker = st.selectbox("Select a ticker symbol:", ["BTC-USD", "ETH-USD", "LTC-USD"])

    date_range = st.date_input("Select a date range:", value=(pd.Timestamp('2022-01-01'), pd.Timestamp.today()))

    short_window = st.slider("Short SMA Window:", min_value=1, max_value=100, value=10)
    long_window = st.slider("Long SMA Window:", min_value=1, max_value=200, value=50)

    prediction_steps = st.number_input("Prediction Steps (Linear Regression):", value=1, min_value=1)

    show_current_price = st.checkbox("Show Current Price")
    show_predicted_price = st.checkbox("Show Predicted Price")
    show_sma_analysis = st.checkbox("Show SMA Analysis")

    if st.button("Predict"):
        if ticker:
            # Predict using linear regression
            predicted_closing_price = predict_price(ticker, date_range[0], date_range[1], steps=prediction_steps)
            
            if show_current_price:
                current_price = fetch_data(ticker, start_date=date_range[0], end_date=date_range[1])[1]['Adj Close']
                st.write(f"Current Price for {ticker}:", current_price)

            # Plot historical and predicted prices
              # Hardcode the date range from 2015 to 2024
               start_date = '2015-01-01'
               end_date = '2024-01-01'  # You can adjust this to today's date or a specific date as needed

# Fetch historical data
               data, _ = fetch_data(ticker, start_date=start_date, end_date=end_date)

# Create a figure for the graph
               fig = go.Figure()

# Plot historical prices
               fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Historical Prices'))

# Plot predicted prices
               fig.add_trace(go.Scatter(x=predicted_closing_price.index, y=predicted_closing_price['Predicted Close'], mode='lines', name='Predicted Prices'))

# Update layout for better visualization
               fig.update_layout(title=f'Historical and Predicted Prices for {ticker}',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=True)

# Display the chart
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

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.sidebar:
        with st.empty():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Accept user input
if prompt := st.sidebar.text_input("Chat with me:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.sidebar:
        with st.empty():
            with st.chat_message("assistant"):
                response = response_generator()
                st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
