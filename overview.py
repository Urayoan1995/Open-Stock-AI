import streamlit as st
import numpy as np
from yahooquery import Ticker
import plotly.express as px
import pandas as pd
import datetime as dt


st.title("Ticker Overview")

# Importing Ticker Symbols from NASDAQ Ticker Screener CSV file
# Only use columns Symbol and Name
all_symbols = pd.read_csv(r"nasdaq_screener_1727459614227.csv",\
                          index_col = "Symbol",\
                          usecols = ["Symbol","Name"])

# Ticker will use selectbox built using Symbols List
ticker = st.selectbox(label = "Ticker Symbol",
                      options = all_symbols.index)
# YahooQuery will be used to extract Stock data from selected ticker
stock = Ticker(ticker)

# HISTORIC STOCK PRICE  --------------------------------------------------
# End all Historic stock price movements on present day
end = dt.datetime.today().date()
# If Leap Year, start period 368 days back
start = end - dt.timedelta(days = 365*5 + 2)
# An extra day is added for at least one Leap Year
# The second extra year is because, when calculating Close % Change, last day will be eliminated when empty cells are dropped
# Rearranging Stock Price History Data
stock_data = stock.history(start = start, end = end, adj_ohlc = True)
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
stock_data.dropna(inplace=True)
# -----------------------------------------------------------------------

# PLOTTING HISTORIC OPEN HIGH LOW & CLOSE DATA -----------------------------------
keys = ["Open", "High", "Low", "Close"]

melted_stock_data = stock_data.melt(id_vars = "Date", value_vars = keys)
melted_stock_data.rename(columns = {"value": "Value", "variable": "Variable"}, inplace=True)
fig = px.line(melted_stock_data, 
              x = "Date", 
              y = "Value", 
              color = "Variable",
              title = all_symbols[all_symbols.index == ticker]["Name"].item(), 
              markers = True)
st.plotly_chart(fig)
# -----------------------------------------------------------------------

# ASSET PROFILE BUILDING ------------------------------------------------
info = stock.asset_profile[ticker]
key_info = {"Key": ["Website", "Country", "Industry", "Sector"],\
            "Info": [info["website"], 
                     info["country"], 
                     info["industry"], 
                     info["sector"]]}

st.dataframe(key_info, hide_index = True, use_container_width= True)
# -----------------------------------------------------------------------

# TABS ------------------------------------------------------------------
description, summary, pricing_data, fundamental_data, ai_analysis = st.tabs(["Description", 
                                                                            "Business Summary", 
                                                                            "Pricing Data", 
                                                                            "Fundamental Data",
                                                                            "AI Analysis"])

# ASSET DESCRIPTION TAB
with description:
    st.subheader("Business Description")
    st.write(info["longBusinessSummary"])

# ASSET SUMMARY TAB
with summary:
    st.subheader("Business Summary")
    summary = stock.summary_detail[ticker]
    summary = {"Key":summary.keys(), "Measurement": summary.values()}
    st.dataframe(summary, hide_index = True, use_container_width= True)

# HISTORIC PRICING DATA TABLE
with pricing_data:
    st.subheader("Historic Pricing Movements")
    st.dataframe(stock_data, hide_index = True, use_container_width = True)
    
        
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
            # Dataframe it created
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

# AI ANALYSIS TAB
with ai_analysis:
    st.write("Perform AI analysis on stock data")
    #results = st.button(label = "Perform Analysis",
    #                    on_click= = ai_stock_analysis)