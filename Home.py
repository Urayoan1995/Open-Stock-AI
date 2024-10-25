import streamlit as st

st.set_page_config(page_title = "Home",
                   page_icon = ":material/home"
                   layout = "wide",
                   initial_sidebar_state="auto")

st.title("Open Stock AI")

st.write("""
         Please select from the menu options in the sidebar to use this program.
         1. Search allows users to look up specific stock companies by ticker symbol. Historical and fundamental data will be displayed, and a combination of machine-learning and AI analysis will be performed.
         2. Trending allows one to look at presently trending stock market companies, as well as the most active ones. 
         3. Help will explain how to use the tools, as well as allowing users to report errors.
         4. About will provide more background information on the project tool and development."""
)