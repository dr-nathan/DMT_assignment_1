# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:02:10 2022

@author: nvs690
"""

#mood on day before
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:39:17 2022

@author: nvs690
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

if __name__ == "__main__":

    data = pd.read_csv("C:/Users/nvs690/surfdrive/Downloads/participant.csv",
                       index_col='time', parse_dates=True)

    participants = data['id'].unique()
    MSE_agg = []
    for part in participants:

        dataset = data[data['id'] == part]

        dataset.drop(columns=['Unnamed: 0', 'id'], inplace=True)
        dataset.index.name = 'date'
        
        predictions=[dataset['mood_value'][i-1] if i>0 else np.mean(dataset['mood_value']) for i,j in enumerate(dataset['mood_value'])]
        
        MSE = mse(list(dataset['mood_value']), predictions)
        
        MSE_agg.append(MSE)
        

    print(f"mean MSE: {np.mean(MSE_agg)}")