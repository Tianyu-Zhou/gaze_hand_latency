import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import math
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn import metrics
import process_data


def process(data, p='right', shift_start=100, shift_end=150, l=1, direction=1):
    X_name_right = ['GazePosX', 'GazePosY', 'GazePosZ', 'CameraPosX', 'CameraPosY', 'CameraPosZ', 'RightConPosX',
                    'RightConPosY', 'RightConPosZ']
    X_name_left = ['GazePosX', 'GazePosY', 'GazePosZ', 'CameraPosX', 'CameraPosY', 'CameraPosZ', 'LeftConPosX',
                   'LeftConPosY', 'LeftConPosZ']
    for dataframe in data:
        df = dataframe.copy()
        name = df.at[1, 'Label']
        result = pd.DataFrame(columns=['PosX', 'PosY', 'PosZ'])
        # window=df.rolling(window=shi)
        if p == 'right':
            temp = ['RightConPosX', 'RightConPosY', 'RightConPosZ']
        else:
            temp = ['LeftConPosX', 'LeftConPosY', 'LeftConPosZ']
        for shi in range(shift_start, shift_end):  # 在区间内寻找最合适的窗口位置，默认100-150
            # 在区间内寻找最合适的窗口大小，默认为1
            for item in temp:
                df[item + '+'] = df[item].shift(-shi)
                if l > 1:
                    for size in range(l):
                        df[item + '+' + str(size)] = df[item].shift(size * direction) #direction决定找窗口大小的方向

            df.dropna(inplace=True)
            if p == 'right':  # 分左右手，选取需要的列
                Y_name = []
                if l > 1:     #如果窗口大小大于一，构造新特征
                    for size in range(l):
                        Y_name.append('RightConPosX+' + str(size))
                        Y_name.append('RightConPosY+' + str(size))
                        Y_name.append('RightConPosZ+' + str(size))
                X = df.loc[:, X_name_right.extend(Y_name)]
                Y = df.loc[:,
                    ['RightConPosX+', 'RightConPosY+', 'RightConPosZ+']]
            else:
                Y_name = []
                if l > 1:
                    for size in range(l):
                        Y_name.append('RightConPosX+' + str(size))
                        Y_name.append('RightConPosY+' + str(size))
                        Y_name.append('RightConPosZ+' + str(size))
                X = df.loc[:, X_name_left.extend(Y_name)]
                Y = df.loc[:,
                    ['LeftConPosX+', 'LeftConPosY+', 'LeftConPosZ+']]
            Y.columns = ['PosX', 'PosY', 'PosZ']


if __name__ == '__main__':
    data = process_data.process()
