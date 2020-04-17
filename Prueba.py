# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:42:13 2020

@author: yingy
"""

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from tensorflow import keras

def plot_series(series, y=None, y_pred=None, title = None, x_label="$t$", y_label="$x(t)$"):
    n_steps = series.shape[0]
    plt.plot(series, ".-", label = 'Past Events')
    if y is not None:
        plt.plot(n_steps, y, "rx", markersize=8, label = 'Actual Future')
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "go", label = 'Predicted Future')
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    if title:
        plt.title(title)
    plt.legend()

def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)
    
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path);

dataframe = pd.read_csv(csv_path)
print('The dataset has {} observations'.format(dataframe.shape[0]))
features = list(dataframe.columns)
features = ('\n'.join(features))
print('The dataset has the following features: \n{}'.format(features))

def window(data, start_index, end_index, history_size, target_size):
    data_list = []
    labels_list = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(data) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data_list.append(np.reshape(data[indices], (history_size, 1)))
        labels_list.append(data[i+target_size])
        
    return np.array(data_list), np.array(labels_list)

temp = dataframe['T (degC)']
temp.index = dataframe['Date Time'] # Change the numerical index for its corresponding date

temp_arr = temp.values # transform from series to numpy

n_train = 110000
temp_mean = temp_arr[:n_train].mean()
temp_std = temp_arr[:n_train].std()
temp_norm = (temp_arr-temp_mean)/temp_std

n_batch = 20 # Number of observations in each batch
n_future = 0 # How many steps ahead from the training data

x_train_temp, y_train_temp = window(temp_norm, 0, n_train, n_batch, n_future)
x_val_temp, y_val_temp     = window(temp_norm, n_train, None, n_batch, n_future)

batch_train = 256
n_buffer = 10000

train_data = tf.data.Dataset.from_tensor_slices((x_train_temp, y_train_temp))
train_data = train_data.cache().shuffle(n_buffer).batch(batch_train).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val_temp, y_val_temp))
val_data = val_data.batch(batch_train).repeat()

for x, y in val_data.take(3):
    plot_series(series = x[0], y=y[0].numpy(), y_pred=tf.math.reduce_mean(x[0]))
    plt.show()