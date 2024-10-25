import pandas as pd
import streamlit as st

st.title("Market Dashboard")
url = "https://finance.yahoo.com/markets/stocks/"

trending, most_active = st.tabs(["Trending ", "Most Active"])

with trending:
    trend_df = pd.read_html(url + "trending/")[0]
    st.dataframe(trend_df, use_container_width = True, hide_index = True)

with most_active:
    active_df = pd.read_html(url + "most-active/")[0]
    st.write(active_df, use_container_width = True, hide_index = True)