import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kmeans(k, data, epsilon):
    dataset = pd.read_csv('zip.train', sep=' ', header=None, usecols=list(range(0, 257)))
    np_array = np.array(dataset)
    num_instances, num_features = np_array.shape

    centroids = np.random.randint(0, num_instances-1, size=10)

    np_array.dtype





if __name__ == '__main__':
    kmeans(12, 'zip.test', 0)