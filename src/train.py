import os
import numpy as np
import open3d
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense


INPUT_PCD_DIR = '/Users/apex/soma_classification/dataset/train/'
OUTPUT_WEIGHT_DIR = '/Users/apex/soma_classification/weight/'
EPOCHS = 20

if __name__ == "__main__":
    '''''''''
    Load file
    '''''''''
    _filenames = os.listdir(INPUT_PCD_DIR)
    filenames = [INPUT_PCD_DIR + fname for fname in _filenames]

    splitnames = []
    for fname in _filenames:
        splitname = os.path.splitext(fname)[0]
        splitnames.append(splitname)
    print('input files(splitnames):', splitnames)

    '''''''''
    Load data
    '''''''''
    train_clouds = np.empty((0, 7))
    for fname in filenames:
        cloud = np.loadtxt(fname, skiprows=1)
        print('cloud shape', cloud.shape)
        # print(cloud)
        train_clouds = np.vstack((train_clouds, cloud))
    print('toatal train shape:', train_clouds.shape)
    # print(train_clouds)

    '''''''''
    Make model
    '''''''''
    N = len(train_clouds)

    model = models.Sequential()
    model.add(Dense(8, input_shape=(N, 6), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())

    '''''''''
    Train model
    '''''''''
    x_train = train_clouds[:, 0:6]
    y_train = train_clouds[:, 6]
    x_train = x_train.reshape(N, -1, 6)
    y_train = y_train.reshape(N, -1, 1)
    print('x_train.shape:', x_train.shape, 'y_train.shape', y_train.shape)
    print(x_train)
    print(y_train)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=10)
    print('Finish training')

    # score = model.evaluate(x_train, y_train)
    # print(score)

    '''''''''
    Save data
    '''''''''
    tf.saved_model.save(model, OUTPUT_WEIGHT_DIR)
