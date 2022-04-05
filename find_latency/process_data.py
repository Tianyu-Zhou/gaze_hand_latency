import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import math
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / np.abs(true))


def print_importance(model, s, name):
    booster = model.booster_
    importance = booster.feature_importance(importance_type='split')
    feature_name = booster.feature_name()
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('feature_importance_' + s + '_' + name + '.csv', index=False)


def predict(data, p, preprocess='', shi=5, m='LGB'):
    MAE, MSE, R2, mape = [], [], [], []
    for dataframe in data:
        df = dataframe.copy()
        name = df.at[1, 'Label']
        result = pd.DataFrame(columns=['PosX', 'PosY', 'PosZ'])
        # window=df.rolling(window=shi)
        for item in ['LeftConPosX', 'LeftConPosY', 'LeftConPosZ', 'RightConPosX', 'RightConPosY', 'RightConPosZ']:
            df[item + '+'] = df[item].shift(-shi)
        df.dropna(inplace=True)
        X = df.loc[:,
            ['GazePosX', 'GazePosY', 'GazePosZ', 'CameraPosX', 'CameraPosY', 'CameraPosZ', 'LeftConPosX', 'LeftConPosY',
             'LeftConPosZ', 'RightConPosX', 'RightConPosY', 'RightConPosZ']]
        Y = df.loc[:,
            ['LeftConPosX+', 'LeftConPosY+', 'LeftConPosZ+', 'RightConPosX+', 'RightConPosY+', 'RightConPosZ+']]
        if p == 'left':
            Y.drop(['RightConPosX+', 'RightConPosY+', 'RightConPosZ+'], axis=1, inplace=True)
        else:
            Y.drop(['LeftConPosX+', 'LeftConPosY+', 'LeftConPosZ+'], axis=1, inplace=True)
        Y.columns = ['PosX', 'PosY', 'PosZ']
        train_hour = math.ceil(0.6 * len(X))
        vali_hour = math.ceil(0.8 * len(X))
        trainX, trainY = X.loc[:train_hour, :], Y.loc[:train_hour, :]
        valiX, valiY = X.loc[train_hour:vali_hour, :], Y.loc[train_hour:vali_hour, :]
        testX, testY = X.loc[vali_hour:, :], Y.loc[vali_hour:, :]
        if m == 'LSTM':
            trainX, valiX, testX = trainX.values, valiX.values, testX.values
        if preprocess == 'scale':
            scale(trainX, copy=False)
            scale(valiX, copy=False)
            scale(testX, copy=False)
        elif preprocess == 'normalize':
            normalize(trainX, copy=False)
            normalize(valiX, copy=False)
            normalize(testX, copy=False)
        temp_mae, temp_mse, temp_r2, temp_mape = ['MAE'], ['MSE'], ['R2'], ['MAPE']
        for s in ['PosX', 'PosY', 'PosZ']:
            tra_Y, va_Y, te_Y = trainY[s].values, valiY[s].values, testY[s].values
            if m == 'LGB':
                model = lgb.LGBMRegressor()
                model.fit(trainX, tra_Y, eval_set=[(valiX, va_Y)], early_stopping_rounds=200)
                result[s] = model.predict(testX)
                print_importance(model, s, name)
            else:
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
                valiX = np.reshape(valiX, (valiX.shape[0], valiX.shape[1], 1))
                testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
                model = Sequential()
                model.add(
                    LSTM(50, activation="relu", input_shape=(trainX.shape[1], trainX.shape[2])))
                model.add(Dense(1))
                model.compile(loss='mae', optimizer='adam')
                # fit network
                history = model.fit(trainX, tra_Y, epochs=100, batch_size=32, validation_data=(valiX, va_Y),
                                    verbose=2, shuffle=False)
                ans = model.predict(testX)
                result[s] = ans.ravel()
            temp_mae.append(metrics.mean_absolute_error(te_Y, result[s]))
            temp_mse.append(metrics.mean_squared_error(te_Y, result[s]))
            temp_r2.append(metrics.r2_score(te_Y, result[s]))
            temp_mape.append(MAPE(te_Y, result[s]))
        MAE.append(temp_mae)
        MSE.append(temp_mse)
        R2.append(temp_r2)
        mape.append(temp_mape)
        result.to_csv('result_' + name + '.csv', index_label='Time')
    result_metric = pd.DataFrame(MAE, columns=['metric', 'PosX', 'PosY', 'PosZ'])
    result_metric = result_metric.append(pd.DataFrame(mape, columns=['metric', 'PosX', 'PosY', 'PosZ']))
    result_metric = result_metric.append(pd.DataFrame(MSE, columns=['metric', 'PosX', 'PosY', 'PosZ']))
    result_metric = result_metric.append(pd.DataFrame(R2, columns=['metric', 'PosX', 'PosY', 'PosZ']))
    result_metric.to_csv('result_metric.csv', index=False)


def process(path='./Label/'):
    data = []
    for file_path in os.listdir(path):
        t = pd.read_csv(path + file_path)
        t.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
        t.drop(['EyePosX', 'EyePosY', 'EyePosZ', 'GazePointRotX', 'GazePointRotY', 'GazePointRotZ', 'CameraRotX',
                'CameraRotY', 'CameraRotZ', 'LeftConRotX', 'LeftConRotY', 'LeftConRotZ', 'RightConRotX', 'RightConRotY',
                'RightConRotZ', 'LeftFootPosX', 'LeftFootPosY', 'LeftFootPosZ', 'LeftFootRotX', 'LeftFootRotY',
                'LeftFootRotZ', 'RightFootPosX', 'RightFootPosY', 'RightFootPosZ', 'RightFootRotX', 'RightFootRotY',
                'RightFootRotZ', 'MiddlePosX', 'MiddlePosY', 'MiddlePosZ', 'MiddleRotX', 'MiddleRotY', 'MiddleRotZ',
                'LeftPupilD', 'RightPupilD', 'ObjectName'], axis=1, inplace=True)
        t['Label'] = os.path.basename(file_path)[:-4]
        data.append(t)
    return data


if __name__ == '__main__':
    data = process()
    predict(data, 'right', 'normalize', shi=100, m='LSTM')
