import streamlit as st
import numpy as np
from yahooquery import Ticker
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import datetime as dt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
import os

st.set_page_config(page_title="Search",
                   layout="wide",
                   initial_sidebar_state="auto")

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

    def plot_data(data, columns, ticker_name):
        fig = go.Figure()
        for column in columns:
            fig.add_trace(go.Line(x = data["Date"], 
                                y = data[column], 
                                name = column,
                                text = data[column]))
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
    fig1 = plot_data(stock_data, keys, all_symbols[all_symbols.index == ticker]["Name"].item())

    st.plotly_chart(fig1)
    st.dataframe(stock_data, hide_index = True, use_container_width = True)
    # -----------------------------------------------------------------------




# TIME SERIES ANALYSIS  ----------------------------------------------------------------

# PROPHET -------------------------------------------------------
with prophet_analysis:
    # TIME SERIES ANALYSIS
    st.header("Time Series Forecasting Using FB Prophet")
    
    if st.button("Execute Prophet"):
        # Selecting the historical closing prices
        
        def make_prophet_model(df):
            prophet_df = df[["Date", "Close"]].copy()
            # Renaming the columns to make the column names fit the formula
            prophet_df.rename(columns = {"Date": "ds", "Close": "y"}, inplace=True)

            prophet_train_df = prophet_df.iloc[:int(prophet_df.shape[0] * 0.7), :]
            prophet_test_df = prophet_df.iloc[int(prophet_df.shape[0] * 0.7):, :]

            # Defining the FB Prophet model
            prophet_model = Prophet(growth = "linear")
            # Fitting the data frame
            prophet_model.fit(prophet_train_df)
            
            # Creating a data frame for model fitting and prediction of closing values
            # We will use it to predict the test data, and also an additional 30 days into the future
            
            prophet_test_pred = prophet_model.predict(prophet_test_df)
            
            prophet_forecast = prophet_model.make_future_dataframe(periods = 225)
            prophet_forecast = prophet_model.predict(prophet_forecast)

            # We compare the test data predictions
            # Error Scores
            prophet_r2 = r2_score(y_true=prophet_test_df["y"], y_pred=prophet_test_pred["yhat"])
            prophet_mae = mean_absolute_error(y_true=prophet_test_df["y"], y_pred=prophet_test_pred["yhat"])
            prophet_mse = (root_mean_squared_error(y_true=prophet_test_df["y"], y_pred=prophet_test_pred["yhat"]))**2
        
            return prophet_model, prophet_forecast, prophet_test_df, prophet_test_pred, prophet_r2, prophet_mae, prophet_mse
        
        prophet_model, prophet_forecast, prophet_test_df, prophet_test_pred, prophet_r2, prophet_mae, prophet_mse = make_prophet_model(stock_data)
        
        time_series_errors = {"Metric": ["R-Squared", "Mean Absolute Error", "Mean Squared Error"],
                            "Value": [prophet_r2, prophet_mae, prophet_mse]}
        
        # Creating a plotly figure of the model forecasts
        fig2 = plot_plotly(prophet_model, 
                        prophet_forecast,
                        trend = True, 
                        xlabel = "Dates",
                        ylabel = "Closing Price")
        fig2.layout.update(title_text = f"{ticker} Forecast Using Prophet")
        
        # Plotting test data comparison
        fig_test = go.Figure()
        fig_test.add_scatter(y = prophet_test_df["y"], name = "Actual", hoverinfo = "y")
        fig_test.add_scatter(y = prophet_test_pred["yhat"], name = "Prediction", hoverinfo = "y")
        fig_test.update_layout(
            title = f"{ticker} Test Data Comparing Prophet Predictions w. Actual Values",
            yaxis = dict(title = "Closing Price"),
            xaxis = dict(title = "Date",
                showticklabels= False))
                
        # Creating a plotly figure of the components of the model
        # fig3 = plot_components_plotly(model, forecast_df)
        
        
        st.plotly_chart(fig2)
        
        st.plotly_chart(fig_test)

        st.markdown("**Model Accuracy Scores**")
        st.dataframe(time_series_errors, hide_index = True, use_container_width = True)
        
        # Only showing the forecast predictions
        st.write(f"Forecast dataset for {ticker}")
        st.dataframe(prophet_forecast,
                    hide_index = True, 
                    use_container_width= True)


# TENSORFLOW -------------------------------------------------------------------

# FUNCTIONS ----------------------------------------------------------------------------------------
def make_dataset(df, num_lags, num_steps, num_features):

    scaler = MinMaxScaler()
    new_df = df.filter(["Close"])
    new_df.rename(columns = {"Close":"x"}, inplace=True)
    
    # Using the preceding 40 days to predict the following 30 days
    num_lags = num_lags
    num_steps = num_steps
    
    num_features = num_features # Univariate model (since we are only using the closing price)


    # Defining the lag steps to use for input
    for i in range(num_lags +1):
        new_df.insert(loc = 1, column = f"t-{i}", value = new_df.iloc[:,0].shift(periods=i))

    # Defining the time steps to predict as output
    for j in range(1,num_steps):
        new_df.insert(loc=num_lags+j+1, column = f"t+{j}", value = new_df.iloc[:,0].shift(periods = -j))

    new_df.dropna(inplace=True)
    # Train/Test Split of 70/30
    X_train = new_df.iloc[:int(new_df.shape[0] * 0.7),  1: num_lags + 1].values
    X_test  = new_df.iloc[ int(new_df.shape[0] * 0.7):, 1: num_lags + 1].values
    y_train = new_df.iloc[:int(new_df.shape[0] * 0.7),  num_lags + 1:].values
    y_test  = new_df.iloc[ int(new_df.shape[0] * 0.7):, num_lags + 1:].values

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    y_train = scaler.fit_transform(y_train)
    y_test  = scaler.fit_transform(y_test)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
    X_test  = X_test.reshape(( X_test.shape[0],  X_test.shape[1],  num_features))

    return scaler, X_train, X_test, y_train, y_test

def build_model(df, num_lags, num_steps, num_features):
    
    scaler, X_train, X_test, y_train, y_test = make_dataset(df, num_lags, num_steps, num_features)
    
    # Defining the number of neurons of the LSTM models
    n_neurons = X_train.shape[1] * X_train.shape[2]
    
    # Model will be based on sequential time series
    model = Sequential()
    
    # Bidirectional Long Short-term Memory
    model.add(Bidirectional(LSTM(units = n_neurons,
                                return_sequences = True,
                                activation = "relu",
                                input_shape = (num_lags, num_features)
                        )))
    model.add(Bidirectional(LSTM(units = n_neurons, 
                                return_sequences = False
                                )))
    # Compress output into a dense layer of size of the number of time steps we are predicting
    model.add(Dense(num_steps))
    
    model.compile(loss = "mse",
                optimizer = "adam"
                )
    
    model.fit(X_train, 
              y_train,
              validation_data = (X_test, y_test),
              epochs = 50,
              verbose = 0
              )
        
    return scaler, model, X_test, y_test

def make_forecast(df, model, num_lags, num_steps, scaler):
    
    # Chossing the last days for prediction
    input = np.array(df.filter(["Close"]).iloc[-lags:])
    input = input.reshape(-1,1)
    input = scaler.fit_transform(input)
    input = input.reshape(1, input.shape[0], -1)
    
    # Making forecast using input and transforming it
    forecast = model.predict(input)
    forecast = scaler.inverse_transform(forecast)
    forecast = forecast[0].tolist()
    
    # Making a copy of the dataframe to add Forecast column
    copy_df = df[["Date","Close"]].copy()
    copy_df["Forecast"] = np.nan

    # We loop through each element of the forecast list
    for i in range(len(forecast)):
    # We skip weekends, since market closes.
        if (dt.datetime.strptime(copy_df.iloc[-1]["Date"],'%Y-%m-%d').weekday() == 4):
            new_row = {"Date": [str((dt.datetime.strptime(copy_df.iloc[-1]["Date"],'%Y-%m-%d') + dt.timedelta(days = 3)).strftime("%Y-%m-%d"))],
                        "Close": [np.nan],
                        "Forecast": [forecast[i]]}
            new_row = pd.DataFrame.from_dict(new_row)
            copy_df = pd.concat([copy_df, new_row], ignore_index = True)
        # All dates involving weekdays are registered
        else:
            new_row = {"Date": [str((dt.datetime.strptime(copy_df.iloc[-1]["Date"], "%Y-%m-%d") + dt.timedelta(days = 1)).strftime("%Y-%m-%d"))],
                        "Close": [np.nan],
                        "Forecast": [forecast[i]]}
            new_row = pd.DataFrame.from_dict(new_row)
            copy_df = pd.concat([copy_df, new_row], ignore_index = True)
    
    return copy_df

# Keras Tab Window
with tf_analysis:
    st.header("Time Series Analysis using Keras and TensorFlow") 
    
    # Users press button to execute Keras model
    if st.button("Execute Keras"):
        lags = 30
        steps = 30
        features = 1
        cwd = os.getcwd()
        if os.path.isdir(f"{cwd}\\KerasModels") == False:
            os.mkdir(f"{cwd}\\KerasModels")

        # First, we check if a trained model already exists
        # If it exists, the model file is loaded
        if os.path.isfile(f"{cwd}\\KerasModels\\{ticker}_Model.keras") == True:
            st.write("Loading model ...")
            model = tf.keras.models.load_model(f"{cwd}\\KerasModels\\{ticker}_Model.keras")
            scaler, X_train, X_test, y_train, y_test = make_dataset(stock_data, lags, steps, features)
            predictions = model.predict(X_test)
            del(X_train)
            del(y_train)
            st.write("Model successfully loaded")
            
        # If the model does not exist, the model is built, trained, and saved in storage
        else:
            st.write("Model does not exist. Building and training model ...")
            scaler, model, X_test, y_test = build_model(stock_data, lags, steps, features)
            predictions = model.predict(X_test)
            model.save(f"{cwd}\\KerasModels\\{ticker}_Model.keras", overwrite=True, zipped=None)
            st.write("Model successfully built and saved")
        
        # In either case, the model is then used to make the forecast
        lstm_forecast_df = make_forecast(stock_data, model, lags, steps, scaler)
        
        # A figure is plotted showing the closing prices and the 30-day forecast
        fig4 = plot_data(lstm_forecast_df, ["Close", "Forecast"], ticker + " Historical Prices w. 30-day Forecast")
        st.plotly_chart(fig4)
                
        # We inverse-transform the y_test and test predictions
        inverse_predictions = scaler.inverse_transform(predictions)
        inverse_y_test = scaler.inverse_transform(y_test)
        
        # We transform the tensor of true and predictes values into lists
        y_vals = inverse_y_test[0].tolist()
        y_pred = inverse_predictions[0].tolist()
        # We will have hundreds of cells, each one with a collection of values determined by the number of steps
        # The last entry in each cell of size of num_steps will be the value of the following day. 
        for i in range(inverse_y_test.shape[0]):
            y_vals.append(inverse_y_test[i][-1])
            y_pred.append(inverse_predictions[i][-1])
        
        fig5 = go.Figure()
        fig5.add_scatter(y = y_vals, name = "Actual", hoverinfo = "y")
        fig5.add_scatter(y = y_pred, name = "Prediction", hoverinfo = "y")
        fig5.update_layout(
            title = f"{ticker} Test Data Comparing Predictions w. Actual Values",
            yaxis = dict(title = "Closing Price"),
            xaxis = dict( title = "Date",
                showticklabels= False))
        
        st.plotly_chart(fig5)
        
        # Accuracy scores
        r2 = r2_score(y_true=y_test, y_pred=predictions)
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        mse = (root_mean_squared_error(y_true=y_test, y_pred=predictions))**2
        
        Keras_time_series_errors = {"Metric": ["R-Squared", "Mean Absolute Error", "Mean Squared Error"],
                                    "Value": [r2, mae, mse]}
        
        st.markdown("**Model Accuracy Scores**")
        st.dataframe(Keras_time_series_errors, 
                     use_container_width=True, 
                     hide_index = True)
        


        
        

# LOWER TABS ------------------------------------------------------------------
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
    st.dataframe(summary_library, hide_index = True, use_container_width= True)
    
        
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
            

