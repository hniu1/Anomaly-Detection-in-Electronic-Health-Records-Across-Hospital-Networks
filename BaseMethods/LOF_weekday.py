#coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from plot_utils import PlotResults1Fig

def LOF_voting(X, window_size, window_step, Thresholds, anomaly_score_threshold):
    # print('LOF start ......')
    df_pred = pd.DataFrame()
    for i, threshold in enumerate(Thresholds):
        # pred = LOF_(X, threshold)
        pred = LOF_SlidingWindow(X, window_size, window_step, threshold, anomaly_score_threshold)
        df_pred['pred_' + str(i)] = pred
        # df_pred['pred_' + str(i)] = df_pred.apply(lambda x: 0 if x['pred_' + str(i)] == 1 else 1, axis=1)

    # df_pred['pred'] = df_pred.apply(lambda x: x['pred_0'] + x['pred_1'] + x['pred_2'], axis=1)
    df_pred['pred'] = df_pred.sum(axis=1)
    return df_pred['pred']

def LOF_(X, threshold):
    n_neighbors = int(round(threshold * len(X)))
    # # fit the model
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    y_pred = clf.fit_predict(X)
    return y_pred

def LOF_SlidingWindow(data, window_size, window_step, threshold, anomaly_score_threshold):
    anomaly_count = np.zeros_like(data)
    total_count = np.zeros_like(data)
    # Compute the number of steps needed to include all data points in at least one window
    n_steps = 1 + (len(data) - window_size) // window_step

    for i in range(n_steps):
        # Determine the start and end indices for the current window
        start_idx = i * window_step
        end_idx = start_idx + window_size

        # Adjust the end index of the last window to include all remaining data points
        if i == n_steps - 1:
            end_idx = len(data)

        window_data = data[start_idx:end_idx]

        # Fit a LOF model to the window data
        k_neighbors = int(round(threshold * len(window_data)))
        lof_model = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=threshold)
        lof_model.fit(window_data.reshape(-1, 1))

        # Predict the anomaly labels for each point in the window
        pred = lof_model.fit_predict(window_data.reshape(-1, 1))
        # Update anomaly_count and total_count for each point in the window
        for j in range(len(window_data)):
            if pred[j] == -1:
                anomaly_count[start_idx + j] += 1
            total_count[start_idx + j] += 1

    anomaly_score = anomaly_count / total_count
    labels = np.where(anomaly_score > anomaly_score_threshold, 1, 0)
    y_pred = list(labels.flatten())
    return y_pred
    # return pd.Series(anomaly_score.flatten())


def LOF(X, x_labels, np_data, path_save, n_neighbors_scaled):
    print('LOF test start ......')
    n_neighbors = [int(round(n * len(X))) for n in n_neighbors_scaled]
    # # fit the model
    clf_0 = LocalOutlierFactor(n_neighbors=n_neighbors[0])
    y_pred_0 = clf_0.fit_predict(X)
    pred_0 = list(y_pred_0)
    clf_1 = LocalOutlierFactor(n_neighbors=n_neighbors[1])
    y_pred_1 = clf_1.fit_predict(X)
    pred_1 = list(y_pred_1)
    clf_2 = LocalOutlierFactor(n_neighbors=n_neighbors[2])
    y_pred_2 = clf_2.fit_predict(X)
    pred_2 = list(y_pred_2)
    raw_data = list(np_data)

    pred_high = []
    pred_mid = []
    pred_low = []
    pred_normal = []
    for id, p_0 in enumerate(pred_0):
        if X[id] == 0:
            label = 0
        else:
            label = p_0 + pred_1[id] + pred_2[id]
        if label == -3:
            pred_high.append(1)
            pred_mid.append(0)
            pred_low.append(0)
            pred_normal.append(1)
        elif label == -1:
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

    # with open(path_save + 'results.csv', 'w') as f:
    #     for id, data in enumerate(raw_data):
    #         f.write(str(data) + ',' + str(pred[id]) + '\n')
    # print('write data into file ', path_save, 'results.csv')

    path_save_file = '/'.join(path_save.split('/')[:-2]) + '/' + path_save.split('/')[-2] + '.png'
    PlotResults1Fig(raw_data, x_labels, pred_high, pred_mid, pred_low, path_save_file)

    df_pred = pd.DataFrame(np.column_stack([pred_high, pred_mid, pred_low, pred_normal]),
                           columns=['pred_high', 'pred_mid', 'pred_low', 'pred_normal'])
    df_pred.to_csv(path_save_file + '.csv', index=False)

    return pred_high, pred_mid, pred_low, pred_normal
