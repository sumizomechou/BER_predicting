#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/3/31 20:48
# @Author: Lianyou Jing
# @File  : BER_predicting.py

import keras
import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
keras.__version__

# from keras.datasets import boston_housing

# (train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

dir = os.getcwd()
dataFile = dir+'\\Unknown_Data_set_all'     # 载入训练数据
data = scio.loadmat(dataFile)
data_all = data['train_data']
BER_target_all = data['BER_target']


permutation = np.random.permutation(data_all.shape[0])  # shape[0]矩阵第一维度的长度 np.random.permutation(x)随机x个数
shuffled_dataset = data_all[permutation, :]
shuffled_labels = BER_target_all[permutation]


train_data = shuffled_dataset[0:69999, :]  # 训练数据集
val_data = shuffled_dataset[70000:70999, :]    # 验证数据集
test_data = shuffled_dataset[71000:89999, :]   # 测试数据集

train_targets = shuffled_labels[0:69999, :]    # 训练目标数据集
val_targets = shuffled_labels[70000:70999, :]  # 验证目标数据集
test_targets = shuffled_labels[71000:89999, :]  # 测试目标数据集

mean = train_data.mean(axis=0)  # axis=0压缩行，对列求平均值
train_data -= mean
std = train_data.std(axis=0)    # axis=0计算每一列的标准差
train_data /= std

val_data -= mean
val_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers

from keras import optimizers


def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation=None))
    adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # adamax = keras.optimizers.Adamax(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # rmsprop = keras.optimizers.RMSprop(lr=3e-3, rho=0.9, epsilon=1e-06)
    # sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=False)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])     # rmsprop
    return model


num_epochs = 1200
batch_size = 256
model = build_model()
# Train the model (in silent mode, verbose=0)

history = model.fit(train_data,
                    train_targets,
                    epochs=num_epochs,
                    batch_size=batch_size, verbose=0,
                    validation_data=(val_data, val_targets))
mse_history = history.history['loss']
mae_history = history.history['mae']
val_mse_history = history.history['val_loss']
val_mae_history = history.history['val_mae']

model.save('BER_Predict_model.h5')

# model = load_model('BER_Predict_model.h5')
yhat = model.predict(test_data, verbose=0)

dataNew = 'C://Users//dcfor//Documents//pythonProject//BEREstimate//Data_Predict.mat'
scio.savemat(dataNew, {'Predict_BER': yhat, 'Real_BER': test_targets})


# Evaluate the model on the validation data
val_mse, val_mae = model.evaluate(test_data, test_targets, verbose=0)
print(val_mse)
print(val_mae)

plt.plot(range(1, len(mse_history) + 1), mse_history, 'bo', label='Training MSE')
plt.plot(range(1, len(val_mse_history) + 1), val_mse_history, 'b', label='Validation MSE')

plt.xlabel('Epochs')
plt.ylabel('Validation MSE')
plt.legend()
plt.show()

plt.plot(range(1, len(mae_history) + 1), mae_history, 'bo', label='Training MAE')
plt.plot(range(1, len(val_mae_history) + 1), val_mae_history, 'b', label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()


#
# # model = load_model('BER_Predict_model.h5')
# model.save('BER_Predict_model.h5')
#
# model = load_model('BER_Predict_model.h5')
# yhat = model.predict(test_data, verbose=0)
#
# dataNew = 'C://Users//Administrator//Desktop//DNN_predict//Data_Predict.mat'
# scio.savemat(dataNew, {'Predict_BER': yhat, 'Real_BER': test_targets })
