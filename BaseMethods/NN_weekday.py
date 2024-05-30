#coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors
import copy
from plot_utils import PlotResults1Fig

def NN_voting(X, window_size, window_step, Thresholds, anomaly_score_threshold, metric = 'euclidean'):
    # print('Nearest Neighbors start ......')
    df_pred = pd.DataFrame()
    for i, threshold in enumerate(Thresholds):
        # pred = NN_(X, threshold, metric)
        pred = NN_SlidingWindow(X, window_size, window_step, threshold, anomaly_score_threshold)
        df_pred['pred_' + str(i)] = pred
    # df_pred['pred'] = df_pred.apply(lambda x: x['pred_0'] + x['pred_1'] + x['pred_2'], axis=1)
    df_pred['pred'] = df_pred.sum(axis=1)
    return df_pred['pred']

def NN_(X, threshold, metric = 'euclidean'):
    n_neighbors = int(round(threshold * len(X)))
    if metric == 'euclidean':
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    elif metric == 'mahalanobis':
        X_train_cov = np.cov(X, rowvar=False)
        X_train_cov = np.reshape(X_train_cov, (-1, 1))
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params={'V': X_train_cov})
    try:
        neigh.fit(X)
        na_distance = neigh.kneighbors(X)[0]
    except:
        return
    lst_distances = []
    for distances in na_distance:
        NN_distance = np.mean(distances)
        lst_distances.append(NN_distance)
    arr_dis = np.asarray(lst_distances, dtype=np.float32)
    mean = np.mean(arr_dis, axis=0)
    std = np.std(arr_dis, axis=0)
    threshold = mean + 1.5 * std
    label = []
    for id, distance in enumerate(lst_distances):
        if X[id] == 0:
            label.append(0)
        else:
            if distance > threshold:
                label.append(1)
            else:
                label.append(0)
    return label

def NN_SlidingWindow(data, window_size, window_step, threshold, anomaly_score_threshold):
    """
    Detect anomalies in time series data using KNN with sliding windows.

    Args:
        data (np.array): A 1D numpy array containing the time series data.
        window_size (int): The size of the sliding window.
        window_step (int): The step size for the sliding window.
        k_neighbors (int): The number of neighbors to use in the KNN algorithm.
        threshold (float): The anomaly detection threshold.

    Returns:
        labels (np.array): A 1D numpy array containing the anomaly labels for each data point.
    """

    anomaly_count = np.zeros_like(data)
    total_count = np.zeros_like(data)
    n_steps = 1 + (len(data) - window_size) // window_step

    for i in range(n_steps):
        start_idx = i * window_step
        end_idx = start_idx + window_size

        # Adjust the end index of the last window to include all remaining data points
        if i == n_steps - 1:
            end_idx = len(data)
        window_data = data[start_idx:end_idx]
        # Fit a KNN model to the window data
        k_neighbors = int(round(threshold * len(window_data)))
        knn_model = NearestNeighbors(n_neighbors=k_neighbors)
        knn_model.fit(window_data.reshape(-1, 1))

        # Compute the distance from each point in the window to its kth neighbor
        distances, _ = knn_model.kneighbors(window_data.reshape(-1, 1))
        kth_distances = distances.mean(axis=1)

        # arr_dis = np.asarray(lst_distances, dtype=np.float32)
        mean = np.mean(kth_distances, axis=0)
        std = np.std(kth_distances, axis=0)
        th = mean + 1.5 * std

        # Update anomaly_count and total_count for each point in the window
        for j in range(len(window_data)):
            if kth_distances[j] > th:
                anomaly_count[start_idx+j] += 1
            total_count[start_idx+j] += 1

    anomaly_score = anomaly_count / total_count
    labels = np.where(anomaly_score > anomaly_score_threshold, 1, 0)
    labels = list(labels.flatten())
    return labels
    # return pd.Series(anomaly_score.flatten())

def NN(X, x_labels, np_data, path_save, n_neighbors_scaled, metric = 'euclidean'):
    print('NN ' + metric + ' start ......')

    n_neighbors = [int(round(n * len(X))) for n in n_neighbors_scaled]

    if metric == 'euclidean':
        neigh_0 = NearestNeighbors(n_neighbors=n_neighbors[0], metric=metric)
        neigh_1 = NearestNeighbors(n_neighbors=n_neighbors[1], metric=metric)
        neigh_2 = NearestNeighbors(n_neighbors=n_neighbors[2], metric=metric)
    elif metric == 'mahalanobis':
        X_train_cov = np.cov(X, rowvar=False)
        X_train_cov = np.reshape(X_train_cov, (-1, 1))
        neigh_0 = NearestNeighbors(n_neighbors=n_neighbors[0], metric=metric, metric_params={'V': X_train_cov})
        neigh_1 = NearestNeighbors(n_neighbors=n_neighbors[1], metric=metric, metric_params={'V': X_train_cov})
        neigh_2 = NearestNeighbors(n_neighbors=n_neighbors[2], metric=metric, metric_params={'V': X_train_cov})
    try:
        neigh_0.fit(X)
        na_distance_0 = neigh_0.kneighbors(X)[0]
        neigh_1.fit(X)
        na_distance_1 = neigh_1.kneighbors(X)[0]
        neigh_2.fit(X)
        na_distance_2 = neigh_2.kneighbors(X)[0]
    except:
        return

    lst_distances_0 = []
    lst_distances_1 = []
    lst_distances_2 = []
    for distances in na_distance_0:
        NN_distance = np.mean(distances)
        lst_distances_0.append(NN_distance)
    for distances in na_distance_1:
        NN_distance = np.mean(distances)
        lst_distances_1.append(NN_distance)
    for distances in na_distance_2:
        NN_distance = np.mean(distances)
        lst_distances_2.append(NN_distance)

    arr_dis_0 = np.asarray(lst_distances_0, dtype=np.float32)
    mean_0 = np.mean(arr_dis_0, axis=0)
    std_0 = np.std(arr_dis_0, axis=0)
    arr_dis_1 = np.asarray(lst_distances_1, dtype=np.float32)
    mean_1 = np.mean(arr_dis_1, axis=0)
    std_1 = np.std(arr_dis_1, axis=0)
    arr_dis_2 = np.asarray(lst_distances_2, dtype=np.float32)
    mean_2 = np.mean(arr_dis_2, axis=0)
    std_2 = np.std(arr_dis_2, axis=0)
    # print('mean: ', mean)
    # print('sd: ', std)
    threshold_0 = mean_0 + 2*std_0
    threshold_1 = mean_1 + 2*std_1
    threshold_2 = mean_2 + 2*std_2

    pred_high = []
    pred_mid = []
    pred_low = []
    pred_normal = []
    for id, distance in enumerate(lst_distances_0):
        if X[id] == 0:
            label = 0
        else:
            if distance > threshold_0:
                label_0 = 1
            else:
                label_0 = 0
            if lst_distances_1[id] > threshold_1:
                label_1 = 1
            else:
                label_1 = 0
            if lst_distances_1[id] > threshold_2:
                label_2 = 1
            else:
                label_2 = 0
            label = label_0 + label_1 + label_2
        if label == 3:
            pred_high.append(1)
            pred_mid.append(0)
            pred_low.append(0)
            pred_normal.append(1)
        elif label == 2:
            pred_high.append(0)
            pred_mid.append(1)
            pred_low.append(0)
            pred_normal.append(1)
        elif label == 1:
            pred_high.append(0)
            pred_mid.append(0)
            pred_low.append(1)
            pred_normal.append(1)
        else:
            pred_high.append(0)
            pred_mid.append(0)
            pred_low.append(0)
            pred_normal.append(0)

    print(str(sum(pred_high)), str(sum(pred_mid)), str(sum(pred_low)), str(sum(pred_normal)))

    raw_data = list(np_data)

    # with open(path_save + metric + '_results.csv', 'w') as f:
    #     for id, data in enumerate(raw_data):
    #         f.write(str(data) + ',' + str(pred[id]) + '\n')
    # print('write data into file ', path_save, 'results.csv')

    path_save_file = '/'.join(path_save.split('/')[:-2]) + '/' + path_save.split('/')[-2]
    PlotResults1Fig(raw_data, x_labels, pred_high, pred_mid, pred_low, path_save_file+'.png')

    # save results: find predict, the number of each prediction
    df_pred = pd.DataFrame(np.column_stack([pred_high, pred_mid, pred_low, pred_normal]),
                           columns=['pred_high', 'pred_mid', 'pred_low', 'pred_normal'])
    df_pred.to_csv(path_save_file + '.csv', index=False)

    return pred_high, pred_mid, pred_low, pred_normal
