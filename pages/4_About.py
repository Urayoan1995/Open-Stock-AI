import streamlit as st

st.set_page_config(page_title="About",
                   layout="wide",
                   initial_sidebar_state="auto")

st.title("About")

st.markdown("""
            *Open Stock AI* is an aimed open-source stock market analysis tool whose development began in 2024. 
            The purpose is to provide an open-source alternative to present proprietary software alternatives, 
            fostering open discussion and sharing of forecasting techniques. Part of the goal of this project is that 
            open sharing of the source code will allow for further enhancement and modifications, both in future updates, and
            by individual users who use the present model as a framework. 
            
            The present neural network model was developed using Keras, and uses bidirectional long short-term memory
            layers. These are trained and used for direct multi-step prediction of the closing prices.
            In simple terms, it uses a sequence of days to predict the possible price for a following sequence of days.
            
            Added to this, the open-source Facebook Prophet model was integrated into the project to provide a comparison
            alternative to the developed neural network model.
            """)

st. write("The repository for this code can be found at [GitHub](https://github.com/Urayoan1995/Open-Stock-AI)")

st.markdown("""**DISCLAIMER**: *Open Stock AI* is not meant to serve as a substitute for expert advice.
             Please consult a professional stockbroker in the case that you wish to participate in the purchase and selling of stocks.""")