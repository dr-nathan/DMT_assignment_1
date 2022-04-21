# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:26:36 2022

@author: nvs690
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("C:/Users/nvs690/surfdrive/Downloads/participant.csv", index_col=0)

# first, arima per pp

participants = dataset['id'].unique()

MSE_matrix = []
for participant in participants:

    part_data = dataset[dataset['id'] == participant]

    series = part_data['mood_value']

    # #check if stationary. If not, replace mood with differenced mood
    if adfuller(series, autolag='AIC')[1] > 0.05:
        part_data['mood_value'] = part_data['mood_value'].diff(periods=1)
        series = part_data['mood_value'].dropna()

    print(f"pvalue of ADF test for stationarity: {adfuller(series, autolag='AIC')[1]}")

    # then, predict per person, with personalized ARIMA
    #test split: 80-20
    trainlen = int(round(0.8*(len(series)), 0))
    testlen = len(series)-trainlen
    train = series[:trainlen]
    test = series[trainlen:]

    model = auto_arima(train, start_p=1, start_q=1,
                       test='adf',       # use adftest to find optimal 'd'
                       max_p=3, max_q=3,  # maximum p and q
                       m=1,              # frequency of series
                       d=None,           # let model determine 'd'
                       seasonal=False,   # No Seasonality
                       start_P=0,
                       D=0,
                       trace=True,
                       error_action='ignore',
                       stepwise=True)

    print(model.summary())

    pred = pd.DataFrame(model.predict(n_periods=testlen), index=test.index)

    # Calculate MSE
    mse = mean_squared_error(pred, test)
    MSE_matrix.append(mse)
    print('mse: ' + str(mse))

    plt.plot(train, label='training')
    plt.plot(test, label='test')
    plt.plot(pred, label='predict')
    plt.legend()
    plt.show()

print(f"mean MSE across participants: {np.mean(MSE_matrix)}")
