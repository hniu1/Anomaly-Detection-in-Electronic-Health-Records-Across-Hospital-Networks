#coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import copy
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize
from plot_utils import PlotResults1Fig

def Gaussian_voting(X, window_size, window_step, Thresholds, anomaly_score_threshold):
    # print('Gaussian start ......')
    df_pred = pd.DataFrame()
    for i, threshold in enumerate(Thresholds):
        pred = UnivariateGaussian_slidingwindow(X, window_size, window_step, threshold, anomaly_score_threshold) # sliding window
        # pred = UnivariateGaussian_(X, threshold)
        df_pred['pred_' + str(i)] = pred

    # df_pred['pred'] = df_pred.apply(lambda x: x['pred_0'] + x['pred_1'] + x['pred_2'], axis=1)
    df_pred['pred'] = df_pred.sum(axis=1)
    return df_pred['pred']

def UnivariateGaussian_(X, threshold):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    p = [scipy.stats.norm(mean, std).cdf(x[0]) for x in X]
    label = []
    for id, cdf in enumerate(p):
        cdf = cdf[0]
        if cdf < threshold or cdf > 1-threshold:
            label.append(1)
        else:
            label.append(0)
    return label

def UnivariateGaussian_slidingwindow(X, window_size, window_step, threshold, anomaly_score_threshold):
    """
        Detect anomalies in time series data using Gaussian with sliding windows.

        Args:
            X (np.array): A 1D numpy array containing the time series data.
            window_size (int): The size of the sliding window.
            window_step (int): The step size for the sliding window.
            threshold (float): The anomaly detection threshold.

        Returns:
            label (list): A list containing the prediction for each data point.
        """

    anomaly_count = np.zeros_like(X)
    total_count = np.zeros_like(X)

    n_steps = 1 + (len(X) - window_size) // window_step

    for i in range(n_steps):
        start_idx = i * window_step
        end_idx = start_idx + window_size

        # Adjust the end index of the last window to include all remaining data points
        if i == n_steps - 1:
            end_idx = len(X)
        window_data = X[start_idx:end_idx]
        mean = np.mean(window_data)
        std = np.std(window_data)

        p = [scipy.stats.norm(mean, std).cdf(x[0]) for x in window_data]
        for j, cdf in enumerate(p):
            if cdf < threshold or cdf > 1-threshold:
                anomaly_count[start_idx+j] += 1
            total_count[start_idx+j] += 1
    anomaly_score = anomaly_count / total_count
    label = np.where(anomaly_score > anomaly_score_threshold, 1, 0)
    label = list(label.flatten())
    return label
    # return pd.Series(anomaly_score.flatten())



def UnivariateGaussian(X, x_labels, np_data, path_save, Thresholds):
    print('Gaussian start ......')

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    p = [scipy.stats.norm(mean, std).cdf(x[0]) for x in X]


    threshold_0, threshold_1, threshold_2 = Thresholds
    pred_high = []
    pred_mid = []
    pred_low = []
    pred_normal = []
    for id, cdf in enumerate(p):
        cdf = cdf[0]
        if cdf < threshold_0 or cdf > 1-threshold_0:
            label_0 = 1
        else:
            label_0 = 0
        if cdf < threshold_1 or cdf > 1-threshold_1:
            label_1 = 1
        else:
            label_1 = 0
        if cdf < threshold_2 or cdf > 1-threshold_2:
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
    # print(str(sum(pred_high)), str(sum(pred_mid)), str(sum(pred_low)))
    raw_data = list(np_data)

    path_save_file = '/'.join(path_save.split('/')[:-2]) + '/' + path_save.split('/')[-2]
    PlotResults1Fig(raw_data, x_labels, pred_high, pred_mid, pred_low, path_save_file + '.png')

    # save results: find predict, the number of each prediction
    df_pred = pd.DataFrame(np.column_stack([pred_high, pred_mid, pred_low, pred_normal]), columns=['pred_high', 'pred_mid', 'pred_low', 'pred_normal'])
    df_pred.to_csv(path_save_file + '.csv', index=False)

    return pred_high, pred_mid, pred_low, pred_normal

