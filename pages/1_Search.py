import streamlit as st
import numpy as np
from yahooquery import Ticker
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import datetime as dt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("Ticker Search and Analysis")

# Importing Ticker Symbols from NASDAQ Ticker Screener CSV file
# Only use columns Symbol and Name
all_symbols = pd.read_csv(r"nasdaq_screener_1727459614227.csv",\
                          index_col = "Symbol",\
                          usecols = ["Symbol","Name"])

# Ticker will use selectbox built using Symbols List
ticker = st.selectbox(label = "Search for Ticker by symbol",
                      options = all_symbols.index)


# YahooQuery will be used to extract Stock data from selected ticker
stock = Ticker(ticker)

# HISTORIC STOCK PRICE  --------------------------------------------------

# End all Historic stock price movements on present day
END = dt.datetime.today().date()
# If Leap Year, start period 368 days back
START = END - dt.timedelta(days = 365*5 +1)

# An extra day is added because, when calculating Close % Change, last day will be eliminated, because % Change will be empty cell
# Rearranging Stock Price History Data
stock_data = stock.history(start = START, end = END, adj_ohlc = True)
stock_data.index = [date.strftime("%Y-%m-%d") for date in stock_data.index.get_level_values(1)]
stock_data.reset_index(inplace=True)
stock_data.rename(columns = {"index": "Date",
                            "open":"Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                            "dividends": "Dividends",
                            "splits": "Splits"},
                inplace=True)
stock_data["% Change"] = stock_data["Close"] / stock_data["Close"].shift(1)*100 -100
# Dropping any rows with empty values
stock_data.dropna(inplace=True)

# Business summary is defined here
# It will be necessary for estimating maximum market capacity for a business
summary = stock.summary_detail[ticker]
summary_library = {"Key":summary.keys(), "Measurement": summary.values()}
# -----------------------------------------------------------------------


keys = ["Open", "High", "Low", "Close"]

history, prophet_analysis, tf_analysis = st.tabs(tabs = ["Historic Data", "Prophet Analysis", "Keras Analysis"])




# PLOTTING HISTORIC OPEN HIGH LOW & CLOSE DATA -----------------------------------


with history:
    

    st.header("Historic Pricing Movements")

    def plot_historic_data(data, columns, ticker_name):
        fig = go.Figure()
        for column in columns:
            fig.add_trace(go.Line(x = stock_data["Date"], 
                                y = stock_data[column], 
                                name = column,
                                text = stock_data[column]))
        fig.update_layout(
            xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                label="1m",
                                step="month",
                                stepmode="backward"),
                            dict(count=6,
                                label="6m",
                                step="month",
                                stepmode="backward"),
                            dict(count=1,
                                label="YTD",
                                step="year",
                                stepmode="todate"),
                            dict(count=1,
                                label="1y",
                                step="year",
                                stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                    type="date"
            )
        )
        fig.layout.update(title_text = ticker_name)
        return fig
    fig1 = plot_historic_data(stock_data, keys, all_symbols[all_symbols.index == ticker]["Name"].item())

    st.plotly_chart(fig1)
    st.dataframe(stock_data, hide_index = True, use_container_width = True)
    # -----------------------------------------------------------------------


# TIME SERIES ANALYSIS  ----------------------------------------------------------------


with prophet_analysis:
    # TIME SERIES ANALYSIS 
    st.header("Time Series Forecasting Using FB Prophet")
    # Selecting the historical closing prices
    df_prophet = stock_data[["Date", "Close"]].copy()
    # Renaming the columns to make the column names fit the formula
    df_prophet.rename(columns = {"Date": "ds", "Close": "y"}, inplace=True)

    # Defining the FB Prophet model
    model = Prophet(growth = "linear")
    # Fitting the data frame
    model.fit(df_prophet)
    # Creating a data frame for model fitting and prediction of closing values
    future_df = model.make_future_dataframe(periods = 365)
    # Forecasting
    forecast_df = model.predict(future_df)
    
    # Error Scores
    r2 = r2_score(y_true=df_prophet['y'], y_pred=forecast_df['yhat'].loc[:len(df_prophet)-1])
    mae = mean_absolute_error(y_true=df_prophet['y'], y_pred=forecast_df['yhat'].loc[:len(df_prophet)-1])
    mse = mean_squared_error(y_true=df_prophet['y'], y_pred=forecast_df['yhat'].loc[:len(df_prophet)-1])
    
    time_series_errors = {"Metric": ["R-Squared", "Mean Absolute Error", "Mean Squared Error"],
                          "Value": [r2, mae, mse]}
    
    # Creating a plotly figure of the model forecasts
    fig2 = plot_plotly(model, 
                    forecast_df,
                    trend = True, 
                    xlabel = "Dates",
                    ylabel = "Closing Price")
    fig2.layout.update(title_text = f"Piece-Wise Linear Model Forecast Results for {ticker}")
    # Creating a plotly figure of the components of the model
    # fig3 = plot_components_plotly(model, forecast_df)
    st.plotly_chart(fig2)

    st.write("Accuracy and Error Scores")
    st.dataframe(time_series_errors, hide_index = True, use_container_width = True)
    
    # Only showing the forecast predictions
    st.write(f"Forecast dataset for {ticker}")
    st.dataframe(forecast_df[forecast_df["ds"] > dt.datetime.today()],
                hide_index = True, 
                use_container_width= True)
    
with tf_analysis:
    st.write("Analysis using Keras and Tensor Flow")    


# TABS ------------------------------------------------------------------
description, summary, fundamental_data = st.tabs(["Description", 
                                                                "Business Summary", 
                                                                "Fundamental Data"])

# ASSET DESCRIPTION TAB
with description:
    info = stock.asset_profile[ticker]
    key_info = {"Key": ["Website", "Country", "Phone","Industry", "Sector"],\
            "Info": [info["website"], 
                    info["country"],
                    info["phone"],
                    info["industry"], 
                    info["sector"]]}
    
    st.subheader("Asset Profile")
    st.dataframe(key_info, hide_index = True, use_container_width= True)

    st.subheader("Business Description")
    st.write(info["longBusinessSummary"])

# ASSET SUMMARY TAB
with summary:
    st.subheader("Business Summary")
    st.dataframe(summary, hide_index = True, use_container_width= True)
    
        
# FUNDAMENTAL DATA TAB -------------------------------
with fundamental_data:    
    
    st.subheader("Annual Balance Sheet")
    balance = stock.balance_sheet(trailing = False)
    # If no statement is available, API will return a string
    if type(balance) == str:
        st.write(balance)
    # If statement is available, API will return a dataframe
    else:
        # Statements might include USD and other currencies, we loop over all unique currency values
        for currency in balance["currencyCode"].unique():
            st.write(f"Statement in {currency} currency")
            # Once a copy of the statement dataframe for a specific currency is defined, we clean it
            balance_copy = balance[balance["currencyCode"]==currency]
            # Transposing makes Symbol the columns, and key names the index
            balance_copy = balance_copy.transpose()
            # Row 0 is the dates of statements, we make that the new column names
            balance_copy.columns = list(balance_copy.iloc[0])
            # Index is not named anything specific
            balance_copy.rename_axis("", inplace=True)
            # We skip first three rows, these are date, period type, and currency code
            balance_copy = balance_copy[3:]
            # We loop over each column, renaming it using just the date value, not the date and time
            for column in balance_copy.columns:
                balance_copy.rename(columns = {column: column.date()}, inplace=True)
            # Dataframe created
            st.dataframe(balance_copy, use_container_width=True)
    
    st.subheader("Annual Cashflow Statement")
    cash = stock.cash_flow(trailing = False)
    if type(cash) == str:
        st.write(cash)
    else:
        for currency in cash["currencyCode"].unique():
            st.write(f"Statement in {currency} currency")
            cash_copy = cash[cash["currencyCode"]==currency]
            cash_copy = cash_copy.transpose()
            cash_copy.columns = list(cash_copy.iloc[0])
            cash_copy.rename_axis("", inplace=True)
            cash_copy = cash_copy[3:]
            for column in cash_copy.columns:
                cash_copy.rename(columns = {column: column.date()}, inplace=True)
            st.dataframe(cash_copy, use_container_width=True)
    
    st.subheader("Annual Income Statement")
    income = stock.income_statement(trailing = False)
    if type(income) == str:
        st.write(income)
    else:
        for currency in income["currencyCode"].unique():
            st.write(f"Statement in {currency} currency")
            income_copy = income[income["currencyCode"]==currency]
            income_copy = income_copy.transpose()
            income_copy.columns = list(income_copy.iloc[0])
            income_copy.rename_axis("", inplace=True)
            income_copy = income_copy[3:]
            for column in income_copy.columns:
                income_copy.rename(columns = {column: column.date()}, inplace=True)
            st.dataframe(income_copy, use_container_width=True)    