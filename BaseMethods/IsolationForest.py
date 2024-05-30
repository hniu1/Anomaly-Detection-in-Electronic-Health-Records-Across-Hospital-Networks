# Import IsolationForest
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# Assume that 13% of the entire data set are anomalies
def IForest(df):
    np_data = np.array(df.Freq)
    X = np.reshape(np_data, (-1, 1)).astype(int)
    outliers_fraction = 0.005
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(X)
    df['IF_pred'] = pd.Series(model.predict(X))
    df['IF_pred'] = df.apply(lambda x: 0 if x['IF_pred']==1 else 1, axis=1)
    return df

def IForest_(df, threshold):
    np_data = np.array(df.value)
    X = np.reshape(np_data, (-1, 1)).astype(int)
    outliers_fraction = threshold
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(X)
    pred = pd.Series(model.predict(X))
    return pred

def IForest_SlidingWindow(df, window_size, window_step, threshold, anomaly_score_threshold):
    np_data = np.array(df.value)
    X = np.reshape(np_data, (-1, 1)).astype(int)
    outliers_fraction = threshold
    model = IsolationForest(contamination=outliers_fraction)

    anomaly_count = np.zeros_like(X)
    total_count = np.zeros_like(X)
    n_steps = 1 + (len(X) - window_size) // window_step

    for i in range(n_steps):
        start_idx = i * window_step
        end_idx = start_idx + window_size

        # Adjust the end index of the last window to include all remaining data points
        if i == n_steps - 1:
            end_idx = len(X)
        window_X = X[start_idx:end_idx]
        model.fit(window_X)
        pred = model.predict(window_X)

        anomaly_count[start_idx:end_idx] += (pred == -1).astype(int).reshape(-1, 1)
        total_count[start_idx:end_idx] += 1

    anomaly_score = anomaly_count / total_count
    labels = np.where(anomaly_score > anomaly_score_threshold, 1, 0)
    pred = pd.Series(labels.flatten())
    return pred
    # return pd.Series(anomaly_score.flatten())

def IF_voting(df, window_size, window_step, Thresholds, anomaly_score_threshold):
    # print('IF start ......')
    df_pred = pd.DataFrame()
    for i, threshold in enumerate(Thresholds):
        # pred = IForest_(df, threshold)
        pred = IForest_SlidingWindow(df, window_size, window_step, threshold, anomaly_score_threshold)
        df_pred['pred_' + str(i)] = pred
        # df_pred['pred_' + str(i)] = df_pred.apply(lambda x: 0 if x['pred_' + str(i)] == 1 else 1, axis=1)

    # df_pred['pred'] = df_pred.apply(lambda x: x['pred_0'] + x['pred_1'] + x['pred_2'], axis=1)
    df_pred['pred'] = df_pred.sum(axis=1)
    return df_pred['pred']
