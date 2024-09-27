import pandas as pd
import streamlit as stp

st.title("Market Dashboard")
url = "https://finance.yahoo.com/markets/stocks/"

trending, most_active = st.tabs(["Trending ", "Most Active"])

with trending:
    trend_df = pd.read_html(url + "trending/")[0]
    st.write(trend_df)

with most_active:
    active_df = pd.read_html(url + "most-active/")[0]
    st.write(active_df)