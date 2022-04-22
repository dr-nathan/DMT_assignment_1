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

        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        X_data = pd.DataFrame(X_scaler.fit_transform(dataset[['appCat.other_value', 'appCat.communication_value',
                                                              'circumplex.valence_value', 'screen_value', 'call_value',
                                                              'sms_value', 'appCat.weather_value',
                                                              'appCat.builtin_value', 'appCat.travel_value', "mood_value"]]))
        Y_data = pd.DataFrame(Y_scaler.fit_transform(dataset[['mood_value']]))

        n_future = 1
        n_past = 3

        X_ = []
        Y_ = []

        for i in range(n_past, X_data.shape[0] - n_future + 1):
            X_.append(X_data.iloc[i - n_past: i, 0: X_data.shape[1]])
            Y_.append(Y_data.iloc[i + n_future - 1: i + n_future, :])

        X_ = np.array(X_)
        Y_ = np.array(Y_)

        # Training Set
        X_train = X_[0: int(round(0.8*len(X_))), :]
        Y_train = Y_[0: int(round(0.8*len(Y_))), :]

        # Validation Set
        X_val = X_[int(round(0.8*len(X_))): int(round(0.9*len(X_))), :]
        Y_val = Y_[int(round(0.8*len(Y_))): int(round(0.9*len(Y_))), :]

        # Test Set
        X_test = X_[int(round(0.9*len(X_))):, :]
        Y_test = Y_[int(round(0.9*len(Y_))):, :]

        # define model
        model = Sequential()
        model.add(LSTM(50, activation='sigmoid', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, Y_train, epochs=50, verbose=0, validation_data=(X_val, Y_val))

        # model.summary()

        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)

        trainPredict = Y_scaler.inverse_transform(trainPredict)
        trainY = Y_scaler.inverse_transform(Y_train.reshape((Y_train.shape[0], 1)))
        testPredict = Y_scaler.inverse_transform(testPredict)
        testY = Y_scaler.inverse_transform(Y_test.reshape((Y_test.shape[0], 1)))

        trainScore = mse(trainY, trainPredict)
        #print('Train Score: %.2f MSE' % (trainScore))
        testScore = mse(testY, testPredict)
        #print('Test Score: %.2f MSE' % (testScore))
        MSE_agg.append(testScore)

        # an_array = np.empty((n_past, 1))

        # an_array[:] = np.NaN

        # dataset['mood_pred'] = np.concatenate((an_array, trainPredict, an_array[:-1], testPredict))
        # plt.plot(dataset[['mood_value']])
        # plt.plot(dataset['mood_pred'])
        # plt.xlabel("timepoint")
        # plt.ylabel("mood value")
        # plt.title(f"LSTM for part. {part} ")
        # plt.legend(['mood', 'prediction'])
        # plt.show()

    print(f"mean MSE: {np.mean(MSE_agg)}")
