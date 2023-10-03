import streamlit as st
from datetime import date

from pathlib import Path
import appdirs as ad

CACHE_DIR = ".cache"
ad.user_cache_dir = lambda *args: CACHE_DIR
Path(CACHE_DIR).mkdir(exist_ok=True)

import yfinance as yf
from prophet.plot import plot_plotly
from prophet import Prophet
from plotly import graph_objs as go


START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

print(TODAY)

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "META", "IBM", "AMZN", "DIS", "TSLA")
selected_stocks = st.selectbox("Select Data Set For Predition", stocks)

n_years = st.slider("Years of Prediction: ", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data")
data = load_data(selected_stocks)
data_load_state.text("Loading Data... done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forcast Data")
st.write(forecast.tail())

st.write("forecast data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

name = st.text("")
name.text("Developed by Parham Esmailzadeh")