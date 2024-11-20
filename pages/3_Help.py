import streamlit as st
import pandas as pd
import os

cwd = os.getcwd()

st.set_page_config(page_title="Help",
                   layout="wide",
                   initial_sidebar_state="auto")

st.title("Help")

search, dashboard, tickers = st.tabs(["Search Feature", "Dashboard Feature", "List of Tickers"])

with search:
    st.markdown("""With the search feature, you are able to search for specific stocks via the ticker symbol. 
             You may scroll down to the desired ticker on the markdown box, or begin writing the symbol in order to begin execution.
             This box is located at the top of the Search window. **An example is shown below**""")
    
    st.image(f"{cwd}/Images/Search.png")
             
    st.write("""
             :blue-background[(If you do not know the ticker symbol of a specific company, you may use the list provided in the 'List of Tickers' tab
             available on the Help window.)]""")
                 
    st.markdown("""
             Once a ticker is selected, the stock data associated with that ticker will be extracted and visualized.
             This includes historical *open*, *high*, *low*, and *closing* price.
             
             The chart you see is interactive, if you click on any of the legend items, you will eliminate these from the chart, 
             meaning that you can choose what you want to see.
             Even more, you may use the bar below the plot to adjust the dates, focusing on more recent data if you so desire.
             **An example of a table filtered to only show a specific subset of pricing data is shown below**.""")
    st.image(f"{cwd}/Images/Example_History.png")    
    
    st.write("""
             Added to this, you can also read a wide array of information on the company, including yearly financial reports.
             This information is on a set of tabs below the initial ones.""")
    st.image(f"{cwd}/Images/Example_Fundamental.png")
    
    st.write("""
             Another note is that any table you see is not only interactive, but downloadable. 
             You may click on the 'Save' icon on any of these to download the table as a CSV file.
             This button is the first from left-to-right of all the icons displayed on the top of a table to the right""")
    st.image(f"{cwd}/Images/Table_Buttons.png")
    
    st.markdown("""
             This search tool provides more than just looking at a stock's data.
             Users can also analyze the stock data and make 'forecasts' using two tools:
             
             1. the open-source [Facebook Prophet](https://facebook.github.io/prophet/) project,
             
             2. and a [Keras](https://keras.io/) artificial neural network model based on bidirectional long short-term memory.""")
             
    st.write("To execute either one, select the tab associated with each option, and click the button to execute this tool.")
    
    st.image(f"{cwd}/Images/Prophet_Button.png")
    st.image(f"{cwd}/Images/Keras_Button.png")
    
    st.markdown("""
             In the case of the Keras model, because neural networks might take a long time to train, an additional feature is added to make its use easier.
             Models are immediately saved in storage upon completion of building and training.
             This is used for two things:
             
            1. If a company's model has not been trained before, a model will be trained and saved for future use.
            2. If you decide to look at the same company on a future date, then the stored model will be loaded.""")
             
    st.markdown("""Both the Prophet and Keras models are used to 'forecast' future instances. 
             Presently, the limit of the prediction is a 30 day window.
             This is due to accuracy limitations present in the models.
             However, you may modify the source code by changing the :blue-background[lag] or :blue-background[steps] variables.""")
    
    
with dashboard:
    st.markdown("""With the dashboard feature, you have two tabs to select between the *Trending* stocks, 
             and the *Most Active* stocks. The tab buttons are shown below.""")
    
    st.image(f"{cwd}/Images/Dashboard.png")
             
    st.markdown("""In either case, you can click on the button that is present to refresh these.
             How frequently you wish to refresh the tables is up to you.
             These tables are interactive, and downloadable. 
             You may simply click on the **download** icon, as seen on the top-right of the table.""")
    st.image(f"{cwd}/Images/Table_Buttons.png")
    
with tickers:
    st.write("The list of registered company tickers was obtained from the [NASDAQ](https://www.nasdaq.com/market-activity/stocks/screener) database.")
    all_symbols = pd.read_csv(f"{cwd}/nasdaq_screener_1727459614227.csv",\
                          usecols = ["Symbol","Name"])
    st.dataframe(all_symbols, hide_index=True, use_container_width=True)