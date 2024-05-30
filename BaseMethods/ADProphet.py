import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from plot_utils import PlotResults1Fig
import logging
import os
logging.getLogger('fbprophet').setLevel(logging.WARNING)

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def fit_predict_model(dataframe, interval_width=0.99, changepoint_range=0.8):
    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=interval_width,
                changepoint_range=changepoint_range)
    with suppress_stdout_stderr():
        m = m.fit(dataframe)

    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop=True)
    # print('Displaying Prophet plot')
    # fig1 = m.plot(forecast)
    # fig1.savefig('_pro.png')
    # fig1.show()
    # plt.close()
    return forecast

def detect_anomalies(forecast):
    forecast = forecast.apply(pd.to_numeric)  # convert all columns of DataFrame
    forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    # forecast['fact'] = df['y']

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1


    # anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper']) / forecast['fact']
    forecasted.loc[forecasted['anomaly'] == -1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact']) / forecast['fact']

    return forecasted

def prophet(df_data, x_labels, np_data, path_save, list_parameter):
    list_preds = []
    for parameter in list_parameter:
        (interval_width, changepoint_range) = parameter
        pred = fit_predict_model(df_data, interval_width, changepoint_range)
        pred = detect_anomalies(pred)
        pred.loc[pred['anomaly'] == -1, 'anomaly'] = 1
        list_pred = list(pd.Series(pred['anomaly']).values[:].astype(int))
        list_preds.append(list_pred)
    # pred.to_csv(path_save + 'results.csv', index=False)
    raw_data = list(np_data)

    pred_high = []
    pred_mid = []
    pred_low = []
    pred_normal = []
    for id, p_0 in enumerate(list_preds[0]):
        if raw_data[id] == 0:
            label = 0
        else:
            label = p_0 + list_preds[1][id] + list_preds[2][id]
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

    path_save_file = '/'.join(path_save.split('/')[:-2]) + '/' + path_save.split('/')[-2]
    PlotResults1Fig(raw_data, x_labels, pred_high, pred_mid, pred_low, path_save_file + '.png')

    df_pred = pd.DataFrame(np.column_stack([pred_high, pred_mid, pred_low, pred_normal]),
                           columns=['pred_high', 'pred_mid', 'pred_low', 'pred_normal'])
    df_pred.to_csv(path_save_file + '.csv', index=False)

    return pred_high, pred_mid, pred_low, pred_normal

def prophet_voting(df_data, list_parameter):
    # print('prophet start ......')

    df_pred = pd.DataFrame()
    for i, p in enumerate(list_parameter):
        pred = prophet_(df_data, p)
        df_pred['pred_' + str(i)] = pred
    # df_pred['pred'] = df_pred.apply(lambda x: x['pred_0'] + x['pred_1'] + x['pred_2'], axis=1)
    df_pred['pred'] = df_pred.sum(axis=1)

    return df_pred['pred']

def prophet_(df_data, parameter):

    (interval_width, changepoint_range) = parameter
    pred = fit_predict_model(df_data, interval_width, changepoint_range)
    pred = detect_anomalies(pred)
    pred.loc[pred['anomaly'] == -1, 'anomaly'] = 1
    list_pred = list(pd.Series(pred['anomaly']).values[:].astype(int))
    return list_pred
