#coding=utf-8

'''
This is a demo for a outlier detection algorithm using committee machine for MedCount data. All used defined parameters are also pointed out in comments.
We used the following algorithms:
1. Nearest Neighbor with Euclidean
2. NN with Maha
3. LOF
4. Gaussian

Author: Haoran Niu
Date: Mar 3, 2023
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import os
import time
from datetime import datetime, timedelta
from scipy.stats import zscore
import matplotlib.pyplot as plt
import warnings
import shutil
from sklearn.metrics import confusion_matrix
import sys
from sklearn import preprocessing
import multiprocessing as mp
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,dir_path + '/BaseMethods/')
from LOF_weekday import LOF, LOF_voting
from NN_weekday import NN, NN_voting
from ADProphet import prophet, prophet_voting
from GaussianAD import UnivariateGaussian, Gaussian_voting
from plot_utils import PlotResults1Fig, Plot7Preds, Plot7PredsUpperBound, PlotCPD
from IsolationForest import IForest, IF_voting
from ResultsFilter import filterAllAnomalies
# from sqlalchemy import create_engine, inspect
# import pyodbc
# os.environ["ODBCINI"] = '/mnt/pfb_hitdata/hithd/Task1_VotingMachine/utils/odbc.ini'

########################
# disable all warnings
########################
warnings.filterwarnings("ignore")

def ReadingTestData(path_data):
    print('reading Test data')
    df_raw = pd.read_csv(path_data)
    df_year = df_raw.loc[df_raw['OrderYr'] == year].reset_index(drop=True)
    df_sta3n = df_year['Sta3n'].value_counts().to_frame().reset_index()
    df_sta3n.columns = ['Sta3n', 'counts']
    sta3ns = df_sta3n['Sta3n'].tolist()
    # sta3n = df_sta3n.iloc[0]['Sta3n']
    # df_data = df_year.loc[df_year['Sta3n'] == sta3n].reset_index(drop=True)
    return df_year, sta3ns

def ReadingPatientCount(path_data, sta3n):
    df_raw = pd.read_csv(path_data)
    df_raw = df_raw.loc[df_raw['Sta3n'] == sta3n].reset_index(drop=True)
    return df_raw

class VotingMachine(object):
    def __init__(self, df_raw, detect_range=0):
        self.df_raw = df_raw
        self.detect_range = detect_range
        '''
        0: weekday
        1: weekend
        2: all
        '''
    
    def Con2datetime(self, x):
        year = str(x['Year'])
        month = str(x['Month'])
        day = str(x['Day'])
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        return year + '-' + month + '-' + day

    def Con2dtobject(self, x):
        year = str(int(x['Year']))
        month = str(int(x['Month']))
        day = str(int(x['Day']))
        datetime_str = month + '/' + day + '/' + year
        dt = datetime.strptime(datetime_str, '%m/%d/%Y')
        return dt

    def WeekDay(self, x):
        year = str(x['Year'])[-2:]
        month = str(x['Month'])
        day = str(x['Day'])
        datetime_str = month + '/' + day + '/' + year
        weekday = datetime.strptime(datetime_str, '%m/%d/%y').weekday()
        return weekday

    def Con2date(self, x):
        year = str(x['Year'])
        month = str(x['Month'])
        day = str(x['Day'])
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        return year[-2:] + '/' + month + '/' + day

    def DiscW2Int(self, row):
        dict_dw = {'1day':1, '1-59days':2, 'Later':3}
        if row['Disc_Window'] != 0:
            return dict_dw[row['Disc_Window']]
        else:
            return row['Disc_Window']

    def DisconInfo(self, x):
        if x['Disc_Window'] == 0:
            return 0
        elif x['Disc_Window'] == 1:
            return 1
        else:
            return 2

    def preprocessing(self, discountinued = False):
        self.df_raw['Disc_Window'].fillna(0, inplace=True)
        self.df_raw = self.df_raw.dropna()
        self.df_raw['Discontinued'] = self.df_raw.apply(lambda row: self.DiscW2Int(row), axis=1)
        for key in ['ServiceSection','DisplayGroupName']:
            self.df_raw[key] = self.df_raw[key].str.lower()
        # self.df_raw['Discontinued'] = self.df_raw.apply(lambda row: self.DisconInfo(row), axis=1)
        if discountinued:
            df_GroupOrder = self.df_raw.groupby(['OrderYr', 'OrderMo', 'OrderDay', 'Sta3n', 'Discontinued'],
                                        as_index=False).agg({'Freq': 'sum'})
            # df_GroupOrder = df_GroupOrder.loc[df_GroupOrder['Discontinued'].isin([1,2])]
        else:
            # self.df_raw = self.df_raw.loc[self.df_raw['Discontinued'].isin([1,2])]
            df_GroupOrder = self.df_raw.groupby(['OrderYr', 'OrderMo', 'OrderDay', 'Sta3n'],
                                                as_index=False).agg({'Freq': 'sum'})
        df_GroupOrder.rename({'OrderYr': 'Year', 'OrderMo': 'Month', 'OrderDay': 'Day'}, axis=1, inplace=True)
        return df_GroupOrder

    def Detector_station(self, granularity, sta3n, path_results, normalize=False, changpoint=False, discountinued=True):
        # the function is used for detection for single station
        # os.makedirs(path_results, exist_ok=True)
        # path_plots = path_results+'/Plots/'
        # os.makedirs(path_plots, exist_ok=True)
        df_GroupOrder = self.preprocessing(discountinued=discountinued)
        if discountinued:
            df_comb = df_GroupOrder[['Discontinued']].value_counts().to_frame().reset_index()
        else:
            df_comb = df_GroupOrder[['Sta3n']].value_counts().to_frame().reset_index()
        # larger than 100
        combs = df_comb.loc[df_comb[0]>100]
        if len(combs) > 0:
            if os.path.exists(path_results) and os.path.isdir(path_results):
                shutil.rmtree(path_results)
            os.makedirs(path_results, exist_ok=True)
            path_plots = path_results+'Plots/'
            os.makedirs(path_plots, exist_ok=True)
        else:
            sys.exit(1)
        # combs.to_csv(path_results + 'test_combs.csv', index = False)
        # combs = df_comb.loc[0:5]
        dict_results = {}
        for ix, comb in combs.iterrows():
            # print("loop {}/{}".format(ix, len(combs)-1))
            if discountinued:
                df_data = df_GroupOrder.loc[(df_GroupOrder.Discontinued==comb.Discontinued)]
            else:
                df_data = df_GroupOrder.loc[(df_GroupOrder.Sta3n==comb.Sta3n)]
            df_Freq = df_data.sort_values(['Year', 'Month', 'Day', 'Sta3n']).reset_index(drop=True)
            start_date = self.Con2dtobject(df_Freq.iloc[0])
            end_date = self.Con2dtobject(df_Freq.iloc[-1])
            df_Freq['Sta3n'] = df_Freq['Sta3n'].astype(str)
            df_Freq['WeekDay'] = df_Freq.apply(lambda x: self.WeekDay(x), axis=1)
            df_Freq['Date'] = df_Freq.apply(lambda x: self.Con2date(x), axis=1)  # used for label in plot
            dict_dw = {0:'Non_disc', 1:'1day', 2:'1-59days', 3:'>=60days'}
            if discountinued:
                label_comb = '(Discontinued_{})'.format(dict_dw[comb.Discontinued])
            else:
                label_comb = 'Discontinued_1_2'

            df_merge_new = df_Freq.copy(deep=True)
            df_Freq = df_merge_new[['Year', 'Month', 'Day', 'Sta3n', 'Date', 'Freq', 'WeekDay']]
            df_sta = df_Freq.drop(['Sta3n'], axis=1)
            df_sta.reset_index(drop=True, inplace=True)
            dt = start_date
            while dt <= end_date:
                wd = dt.weekday()
                if self.detect_range == 0:
                    if wd >= 5:
                        dt += timedelta(days=1)
                        continue
                elif self.detect_range == 1:
                    if wd < 5:
                        dt += timedelta(days=1)
                        continue
                dt_str = dt.strftime("%y/%m/%d")
                if dt_str in df_sta['Date'].values:
                    dt += timedelta(days=1)
                    continue
                else:
                    dt_str_long = dt.strftime("%Y/%m/%d")
                    [year, month, day] = dt_str_long.split('/')
                    dt += timedelta(days=1)
                    new_row = {'Year': int(year), 'Month': int(month), 'Day': int(day), 'Freq': 0, 'WeekDay': wd,
                                'Date': dt_str}
                    df_sta = df_sta.append(new_row, ignore_index=True)
            df_sta = df_sta.sort_values(['Year', 'Month', 'Day'])
            df_sta.reset_index(drop=True, inplace=True)
            if granularity == 'weekly':
                df_weekly = pd.DataFrame()
                for index, row in df_sta.iterrows():
                    if index % 7 == 0:
                        week = index // 7
                        if week > 0:
                            df_weekly = df_weekly.append({'week': week, 'Freq': sum_week,
                                                            'Year': str(row['Year']), 'Month': str(row['Month']),
                                                            'Day': str(row['Day']), 'Date': row['Date']}, ignore_index=True)
                        sum_week = 0.0
                    sum_week += row['Freq']
                df_sta = df_weekly
            df_sta['Freq'] = df_sta['Freq'].astype(float)
            df_sta.reset_index(drop=True, inplace=True)
            # holidays
            cal = calendar()
            dts = df_sta[['Year', 'Month', 'Day']]
            datetime = pd.to_datetime(dts)
            holidays = cal.holidays(start=datetime.min(),end=datetime.max())
            df_sta['Holiday']= datetime.isin(holidays)
            df_sta = df_sta.loc[df_sta['Holiday'] == False].drop('Holiday', axis=1)
            df_sta.reset_index(drop=True, inplace=True)
            if self.detect_range == 0:
                df_sta = df_sta.loc[df_sta['WeekDay']<5]
                df_sta.reset_index(drop=True, inplace=True)
            if not changpoint:
                df_pred = self.VM_upperBound(df_sta, granularity, normalize)
                # df_pred = self.PostProcessing(df_pred)
                Plot7PredsUpperBound(df_pred, path_plots+'Plots_{}_'.format(label_comb))
                df_anomalies = filterAllAnomalies(df_pred)
            else:
                df_pred = self.ChangePointDetector(df_sta, path_results, normalize)
                df_anomalies = filterAllAnomalies(df_pred)
                df_anomalies.to_csv(path_results + 'CPD.csv', index=False)
            
            # add anomalies to results
            for index, row in df_anomalies.iterrows():
                date = row['timestamp']
                count = int(row['value'])
                # spc = row['SPC']
                VotingComm = row['voting']
                if VotingComm:
                    if date not in dict_results:
                        dict_results[date] = {}
                        dict_results[date]['VotingComm'] = ''
                        # dict_results[date]['VotingComm'].append('{}: {}'.format(label_comb, count))
                    dict_results[date]['VotingComm'] += '{}: {}'.format(label_comb, count) + '\n'

        df_results = pd.DataFrame.from_dict(dict_results, orient='index')
        df_results.to_csv(path_results + 'Results.csv', index=True)

        # return df_results


    def ChangePointDetector(self, df_sta, path_save, normalize=False):
        df_X = df_sta.copy(deep=True)
        df_X['timestamp'] = df_X.apply(lambda row: '-'.join([str(row['Year']), str(row['Month']), str(row['Day'])]),
                                       axis=1)
        df_X = df_X[['timestamp', 'Freq']]
        df_X.rename(columns={'Freq': 'value'}, inplace=True)
        df_pred = df_X.copy(deep=True)
        # normalize
        if normalize:
            x = np.array(df_X.value)
            x = np.reshape(x, (-1, 1)).astype(int)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df_X.value = x_scaled
        df_freq = df_X['value']
        np_data = pd.Series(df_freq).values[:].astype(int)
        X = np.reshape(np_data, (-1, 1)).astype(int)

        df_pred_cpd = ChangePointDetection(X, np_data)
        # append changepoint Detection results
        df_pred = df_pred.join(df_pred_cpd)
        PlotCPD(df_pred, path_save)
        return df_pred

    def VM_upperBound(self, df_sta, granularity, normalize=False):
        df_X = df_sta.copy(deep=True)
        df_X['timestamp'] = df_X.apply(lambda row: '-'.join([str(row['Year']), str(row['Month']), str(row['Day'])]),
                                       axis=1)
        df_X = df_X[['timestamp', 'Freq']]
        df_X.rename(columns={'Freq': 'value'}, inplace=True)
        df_pred = df_X.copy(deep=True)

        # # normalize
        # if normalize:
        #     x = np.array(df_X.value)
        #     x = np.reshape(x, (-1, 1)).astype(int)
        #     # min_max_scaler = preprocessing.MinMaxScaler()
        #     # x_scaled = min_max_scaler.fit_transform(x)
        #     x_scaled = zscore(x)
        #     df_X.value = x_scaled
        df_freq = df_X['value']
        np_data = pd.Series(df_freq).values[:].astype(int)
        X = np.reshape(np_data, (-1, 1)).astype(int)

        window_size = round(len(X) * window_size_ratio)
        window_step = round(window_size * window_step_ratio)

        # IF
        Thresholds = [0.05] if not discountinued else [0.05] # outliers_fraction
        pred_IF = IF_voting(df_X, window_size, window_step, Thresholds, anomaly_score_threshold)
        df_pred['IF'] = pred_IF
        df_pred['IF'] = df_pred.apply(lambda row: 1 if row['IF'] > 0 else 0, axis=1)
        # df_pred['IF'] = 0

        # Gaussian
        Thresholds = [0.03] if not discountinued else [0.03]
        pred_Gaussian = Gaussian_voting(X, window_size, window_step, Thresholds, anomaly_score_threshold)
        df_pred['G'] = pred_Gaussian
        df_pred['G'] = df_pred.apply(lambda row: 1 if row['G'] > 0 else 0, axis=1)

        # # algorithm KNN
        metric = 'euclidean'
        n_neighbors = [0.3] if not discountinued else [0.1]
        pred_NNE = NN_voting(X, n_neighbors, metric, window_size, window_step, anomaly_score_threshold)
        df_pred['KNN'] = pred_NNE
        df_pred['KNN'] = df_pred.apply(lambda row: 1 if row['KNN'] > 0 else 0, axis=1)

        # algorithm LOF
        n_neighbors = [0.1] if not discountinued else  [0.1]
        pred_LOF = LOF_voting(X, window_size, window_step, n_neighbors, anomaly_score_threshold)
        df_pred['LOF'] = pred_LOF
        df_pred['LOF'] = df_pred.apply(lambda row: 1 if row['LOF'] > 0 else 0, axis=1)

        # algorithm Prophet
        df_sta['ds'] = df_sta.apply(lambda x: self.Con2datetime(x), axis=1)
        df_prophet = df_sta[['ds', 'Freq']]
        df_prophet.columns = ['ds', 'y']
        list_parameter = [(0.85, 0.8)]  if not discountinued else [(0.90, 0.8)]
        pred_prophet = prophet_voting(df_prophet, list_parameter)
        df_pred['Prophet'] = pred_prophet
        df_pred['Prophet'] = df_pred.apply(lambda row: 1 if row['Prophet'] > 0 else 0, axis=1)

        df_pred['voting'] = df_pred.apply(lambda row: 1 if sum(row[2:])>num_voting else 0, axis = 1)

        return df_pred

    def PostProcessing(self, df):
        # less than average equal to 0
        # df.loc[df['value'] < df['value'].mean(), ['IQR', 'IF', 'G', 'NNE', 'voting']] = 0
        # zero count will not be considered
        df.loc[df['value'] == 0, ['IQR', 'IF', 'G', 'NNE', 'voting']] = 0
        return df

def process_sta3n(args):
    sta3n, year, df_year = args
    print(f'station {sta3n}')
    df_data = df_year.loc[df_year['Sta3n'] == sta3n].reset_index(drop=True)
    path_results = path_abs + 'results/{}_allsta3n_new/Year_{}/Station_{}/'.format(domain, year, sta3n)
    VM = VotingMachine(df_data)
    path_sub = 'Station_{}_allorder/'.format(sta3n) if not discountinued else 'Station_{}_all_DW/'.format(sta3n)
    VM.Detector_station(granularity, sta3n, path_results+path_sub, normalize, changpoint=changpoint, discountinued=discountinued)

if __name__ == '__main__':
    domain = 'Con'
    num_voting = 2 # voting strategy: pred larger than 2 is anomaly
    window_size_ratio = 0.1 # sliding window size ratio
    window_step_ratio = 0.25 # sliding window step ratio in each window
    anomaly_score_threshold = 0.5
    path_abs = '../'
    # path_abs = '/mnt/pfb_hitdata/hithd/Task1_VotingMachine/'
    path_data = path_abs + 'data/{}/{}CountAll.csv'.format(domain, domain)
    normalize = False
    changpoint = False
    discountinued = True
    granularity = 'daily'
    years = [2019]

    for year in years:
        df_year, sta3ns = ReadingTestData(path_data)
        with mp.Pool(processes=4) as p:
            with tqdm(total=len(sta3ns)) as pbar:
                for _ in p.imap_unordered(process_sta3n, [(sta3n, year, df_year) for sta3n in sta3ns[:1]]):
                    pbar.update()
    print('test finished!!')


