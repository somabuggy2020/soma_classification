import os
import numpy as np
import open3d
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

INPUT_PCD_DIR = '/Users/apex/soma_classification/dataset/train/'
INPUT_WEIGHT_DIR = '/Users/apex/soma_classification/weight/'
OUTPUT_PCD_DIR = '/Users/apex/soma_classification/output/'

if __name__ == "__main__":
    '''''''''
    Load file, model
    '''''''''
    _filenames = os.listdir(INPUT_PCD_DIR)
    filenames = [INPUT_PCD_DIR + fname for fname in _filenames]
    splitnames = []
    for fname in filenames:
        splitname = os.path.splitext(fname)[0]
        splitnames.append(splitname)
    print('input files(splitnames):', splitnames)

    model = models.load_model(INPUT_WEIGHT_DIR)
    print(model.summary())

    predicted_ground = np.empty((0, 6))
    predicted_slope = np.empty((0, 6))

    '''''''''
    Load data
    '''''''''
    clouds = np.empty((0, 7))
    for fname in filenames:
        pcd = np.loadtxt(fname, skiprows=1)
        print('pcd shape', pcd.shape)
        # print(pcd)
        clouds = np.vstack((clouds, pcd))
        test_clouds = clouds[:, 0:6]
    test_clouds = test_clouds.reshape(len(test_clouds), -1, 6)
    print('total test shape:', test_clouds.shape)
    print(test_clouds)

    '''''''''
    Predict
    '''''''''
    raw_predictions = model.predict(test_clouds)
    predictions = np.round(raw_predictions)
    print('predictions', predictions.shape)
    i = 0
    for pd in predictions:
        if pd == 0:
            predicted_ground = np.vstack(
                (predicted_ground, test_clouds[i]))
        else:
            predicted_slope = np.vstack(
                (predicted_slope, test_clouds[i]))
        i = i + 1

    print('ground', predicted_ground.shape, 'slope', predicted_slope.shape)

    '''''''''
    Export pcd
    '''''''''
    ground_pcd = open3d.geometry.PointCloud()
    slope_pcd = open3d.geometry.PointCloud()

    ground_pcd.points = open3d.utility.Vector3dVector(predicted_ground[:, 0:3])
    slope_pcd.points = open3d.utility.Vector3dVector(predicted_slope[:, 0:3])

    open3d.io.write_point_cloud(
        OUTPUT_PCD_DIR + 'output_slope.pcd', slope_pcd, write_ascii=True)
    open3d.io.write_point_cloud(
        OUTPUT_PCD_DIR + 'output_ground.pcd', ground_pcd, write_ascii=True)
