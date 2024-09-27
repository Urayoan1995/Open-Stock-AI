import streamlit as st
import numpy as np
from yahooquery import Ticker
import plotly.express as px
import pandas as pd


st.title("Ticker Overview")


all_symbols = pd.read_csv(r"nasdaq_screener_1727459614227.csv",\
                          index_col = False,\
                          usecols = ["Symbol","Name"])

st.sidebar.write("All Market Ticker Symbols")
st.sidebar.table(all_symbols, hide_index = True)

ticker = st.text_input("Ticker Symbol")
stock = Ticker(ticker)

end = dt.datetime.today().date()
if end.year % 4 == 0:
    start = end - dt.timedelta(days = 368)
else:
    start = end - dt.timedelta(days = 367)

stock_data = stock.history(start = start, end = end)
stock_data.index = [date.strftime("%Y-%m-%d") for date in stock_data.index.get_level_values(1)]
stock_data.rename_axis("Date", inplace=True)
stock_data.rename(columns = {"open":"Open",
                            "high": "High",
                            "close": "Close",
                            "close": "Close",
                            "volume": "Volume",
                            "adjclose": "Adj Close",
                            "dividends": "Dividends",
                            "splits": "Splits"},
                  inplace=True)
stock_data["% Change"] = stock_data["Close"] / stock_data["Close"].shift(1)*100 -100
stock_data.dropna(inplace=True)

fig = px.line(stock_data, 
            x = stock_data.index, 
            y = stock_data[["Close","Adj Close"]], 
            title = ticker, 
            markers = True, 
            labels = ["Close", "Adjusted Close"])
st.plotly_chart(fig)

info = stock.asset_profile[ticker]

key_info = {"Key": ["Website", "Country", "Industry", "Sector"],\
            "Info": [info["website"], 
                     info["country"], 
                     info["industry"], 
                     info["sector"]]}

st.table(pd.DataFrame.from_dict(key_info).set_index("Key"))

pricing_data, fundamental_data = st.tabs(["Pricing Data", "Fundamental Data"])

with pricing_data:
    st.header("Movements")
    
    std = round(np.std(stock_data["% Change"])*np.sqrt(252), 4) 
    annual_return = round(stock_data["% Change"].mean()*252, 4)
    risk_adj_return = round(annual_return / std, 4)
    
    movement_metrics = {"Metric": ["Standard Deviation", "Annual Return", "Risk Adj. Return"],\
                            "% Value": [std, annual_return, risk_adj_return]}
    
    st.write(pd.DataFrame.from_dict(movement_metrics).set_index("Metric"))
    st.write(stock_data)
    
    
# Fundamental Data Tab -------------------------------
with fundamental_data:    
    
    st.subheader("Annual Balance Sheet")
    balance = stock.balance_sheet()
    balance = balance.transpose()
    balance.columns = list(balance.iloc[0])
    balance.rename_axis("", inplace=True)
    balance = balance[3:]
    st.table(balance)
    
    st.subheader("Annual Cashflow Statement")
    cash = stock.cash_flow()
    cash = cash.transpose()
    cash.columns = list(cash.iloc[0])
    cash.rename_axis("", inplace=True)
    cash = cash[3:]
    st.table(cash)
    
    st.subheader("Annual Income Statement")
    income = stock.income_statement()
    income = income.transpose()
    income.columns = list(income.iloc[0])
    income.rename_axis("", inplace=True)
    income = income[3:]
    st.table(income)
