from scipy.stats import boxcox 
from sklearn import preprocessing
import statsmodels.api as sm
import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

warnings.filterwarnings('ignore')

st.header("***Forecasting gold price for upcomming 30 days***")

days = st.slider('No of days forecast required (max of 100 days)', min_value = 1, max_value = 100)

st.write('This model is forecasts Gold Price for upcomming days')


gold_price=pd.read_csv('Gold_data.csv',parse_dates=['date'])
#gold_price=gold_price.set_index('date' , drop=True)

## Model
#import statsmodels.api as sm
#from statsmodels.tsa.holtwinters import ExponentialSmoothing

FINAL_ARIMA_MODEL= ARIMA(gold_price['price'], order=(31,1,31)).fit()

model_fore = FINAL_ARIMA_MODEL.forecast(30)

model_fore=pd.DataFrame(model_fore)
model_fore.columns=[('forecast')]

    
 ##plot
#st.line_chart(data=SARIMA_fore, width=0, height=0, use_container_width=True)

bt=st.button("Forecast")
if bt is True:
    st.write(model_fore)
    plt.figure(figsize=(16,8))
    plt.plot(model_fore['forecast'])
    st.pyplot()

