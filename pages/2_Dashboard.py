import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard",
                   layout="wide",
                   initial_sidebar_state="auto")

st.title("Market Dashboard")
url = "https://finance.yahoo.com/markets/stocks/"

trending, most_active = st.tabs(["Trending ", "Most Active"])

with trending:
    if st.button("Refresh Trending Stocks"):
        df = pd.read_html(url + "trending/")[0]
        df.drop(columns = [df.columns[2], df.columns[-1]], inplace=True)
        st.dataframe(df, use_container_width = True, hide_index = True)

with most_active:
    if st.button("Refresh Active Stocks"):
        df = pd.read_html(url + "most-active/")[0]
        df.drop(columns = [df.columns[2], df.columns[-1]], inplace=True)
        st.write(df, use_container_width = True, hide_index = True)