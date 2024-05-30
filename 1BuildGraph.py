'''
Start to build anomaly graph based on the results of voting machine.
* read data into dictionary
* convert dictionary data into graph format

Author: Haoran Niu
Date: Mar 9, 2023
'''

import os
import sys
import pandas as pd
import tqdm
import pickle
import subprocess
import networkx as nx
from matplotlib import pyplot as plt
import glob
from datetime import datetime



def HistPlot(path_save):
    num_bin = 20
    df = pd.read_csv(path_save + 'Graph.tsv',
                     sep="\t", header=None, names=['V0', 'V1', 'Edge'], skiprows=1)
    df['Edge_scale'] = df['Edge']/df['Edge'].max()
    fig = plt.figure(figsize=(6.5, 5))
    plt.hist(df['Edge_scale'], bins=num_bin, range=(0, 1), alpha=0.8, edgecolor='black', linewidth=1.0)
    plt.yscale('log')
    plt.title('Histogram of Scaled Edge Weight')
    plt.xticks([i / 10.0 for i in range(0, 11, 1)])
    plt.xlabel(f"Scaled Edge Weight (bins={num_bin})")

    plt.tight_layout()
    plt.savefig(path_save + 'edge_weight_histogram.png', dpi=300)
    plt.show()
    print()

def ReadResults(path, sta3n, dict_pred):
    '''
    Aug:
        Path: path of results
        sta3n: station number
        dict_pred: the dictionary store all the results
    :return: dict_pred
    '''
    df_results = pd.read_csv(path+'/Results.csv', index_col=False)
    df_results = df_results.rename(columns={df_results.columns[0]: 'Date'})

    for i, row in df_results.iterrows():
        date = row['Date']
        results = row['VotingComm']
        anomalies = results.split('\n')[:-1]
        if date not in dict_pred:
            dict_pred[date] = {}
        for anomaly in anomalies:
            signal = anomaly.split(':')[0]
            if '_normalized' in path:
                signal += '-normalized'
            if signal not in dict_pred[date]:
                dict_pred[date][signal] = []
            dict_pred[date][signal].append(sta3n)

    return dict_pred

def ReadAnomalies():
    '''
    Aug:

    :return: dictionary dic_pred: {key: date, value: dictionary signal: {key: signal, value: sta3n}}
    '''
    dict_pred = {}
    icon = "➡️"

    for dir_year in glob.glob(path_results + 'Year_*'):
        year = dir_year.split('_')[-1]
        if year not in dict_pred.keys():
            dict_pred[year] = {}
        for dir_sta3n in tqdm.tqdm(glob.glob(dir_year + '/Station_*'), desc=f'Year: {year} {icon}'):
            station = dir_sta3n.split('_')[-1]
            for dir_signal in glob.glob(dir_sta3n + '/[!.]*'):
                if '_normalized' in dir_signal:
                    continue
                dict_pred[year] = ReadResults(dir_signal, station, dict_pred[year])

    # open a file for binary writing
    with open(path_graph + "/Summary_Anomalies.pickle", "wb") as f:
        # write the dictionary to the file using pickle.dump
        pickle.dump(dict_pred, f)
    return

def ConvertResults2Graph():
    '''
    Aug:
    :return: graph file: DIMACS format using tsv
    '''
    dict_graph = {}
    '''
    key: edge; value: count
    '''
    with open(path_graph + "/Summary_Anomalies.pickle", "rb") as f:
        # load the dictionary from the file using pickle.load
        dict_pred = pickle.load(f)
    for year in dict_pred:
        if target_year != 'all' and target_year != year:
            continue
        for date in tqdm.tqdm(dict_pred[year], total=len(dict_pred[year]), desc=f"Year {year}:"):
            for signal in dict_pred[year][date]:
                list_sta3n = list(set(dict_pred[year][date][signal]))
                if len(list_sta3n) > 1: # more than 1 stations
                    for i in range(len(list_sta3n)-1):
                        for j in range(i+1, len(list_sta3n)):
                            edge = frozenset([list_sta3n[i], list_sta3n[j]])
                            if len(edge)!=2:
                                print(f'false in signal {signal} in date {date}')
                                sys.exit(1)
                            if edge not in dict_graph:
                                dict_graph[edge] = 0
                            dict_graph[edge] += 1
    # open a file for binary writing
    with open(path_save + "dict_graph.pickle", "wb") as f:
        # write the dictionary to the file using pickle.dump
        pickle.dump(dict_graph, f)

    # write file into DIMACS format
    # create a DataFrame from the dictionary
    df = pd.DataFrame.from_records(list(dict_graph.items()), columns=['Set', 'Value'])

    # split the set column into separate columns
    df = pd.concat([df['Set'].apply(lambda x: pd.Series(list(x))), df.drop('Set', axis=1)], axis=1)
    df.columns = ['V0', 'V1', 'Edge']
    df = df.sort_values(by='Edge', ascending=False)
    num_nodes = len(set(df['V0']).union(set(df['V1'])))
    num_edges = len(df)

    # create a string with the first line of the file
    first_line = f"{num_nodes}\t{num_edges}\n"

    # save the DataFrame to a TSV file without a header
    with open(path_save+"Graph.tsv", "w") as f:
        f.write(first_line)
        df.to_csv(f, sep="\t", header=False, index=False)

    return

def ScaledWeightGraph(path_save):
    '''
    Translate edge weight into scaled edge weight.
    :return:
    '''
    df = pd.read_csv(path_save + 'Graph.tsv',
                     sep="\t", header=None, names=['V0', 'V1', 'Edge'], skiprows=1)
    df['Edge'] = df['Edge']/df['Edge'].max()
    num_nodes = len(set(df['V0']).union(set(df['V1'])))
    num_edges = len(df)

    # create a string with the first line of the file
    first_line = f"{num_nodes}\t{num_edges}\n"

    # save the DataFrame to a TSV file without a header
    with open(path_save + "Graph_scaled.tsv", "w") as f:
        f.write(first_line)
        df.to_csv(f, sep="\t", header=False, index=False)
    return

def FindThreshold_spectral():
    df = pd.read_csv(path_save + 'Graph_scaled_iterative.txt', sep="\t")
    # threshold = df.loc[df['2nd-eigenvalue'].idxmin()]['threshold']
    # min_eigen = df['2nd-eigenvalue'].min()
    threshold = 0.78
    min_eigen = df[df['threshold']==threshold]['2nd-eigenvalue']

    df_plot = df[df['threshold'] >= 0.1]
    # plot the results
    plt.rcParams.update({'font.family': 'sans-serif'})
    fig = plt.figure(figsize=(7,5))
    thresholds = df_plot['threshold'].to_list()
    plt.plot(thresholds, df_plot['2nd-eigenvalue'], 'orangered', marker='.', markersize=10, linewidth=2.0)
    plt.xlabel('Threshold', fontsize=12, weight='bold')
    plt.ylabel('Algebraic connectivity (2nd-eigenvalue)', fontsize=12, weight='bold')
    bottom = plt.gca().get_ylim()[0]
    plt.text(threshold, (min_eigen+bottom)/2, str(threshold), ha='center', va='center')
    plt.plot(threshold, min_eigen, marker='.', markersize=10, color='black')
    plt.title('Spectral Thresholding ', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(path_save + 'thresholding.png', dpi=300)
    plt.show()

    # generate new graph after thresholding
    df = pd.read_csv(path_save + 'Graph_scaled.tsv',
                     sep="\t", header=None, names=['V0', 'V1', 'Edge'], skiprows=1)
    # df['Edge'] = df['Edge'] / df['Edge'].max()
    df_th = df[df['Edge'] > threshold][['V0','V1']]
    num_nodes = len(set(df_th['V0']).union(set(df_th['V1'])))
    num_edges = len(df_th)

    # create a string with the first line of the file
    first_line = f"{num_nodes}\t{num_edges}\n"

    # save the DataFrame to a TSV file without a header
    with open(path_save + "Graph_threshold.tsv", "w") as f:
        f.write(first_line)
        df_th.to_csv(f, sep="\t", header=False, index=False)
    return

def PlotParaclique():
    path_paraclique = path_save + 'paraclique'
    os.makedirs(path_paraclique, exist_ok=True)

    list_parac = []
    with open(path_save + "paraclique.txt", "r") as f:
        for line in f:
            paraclique = line.strip().split('\t')
            list_parac.append(paraclique)

    df = pd.read_csv(path_save + 'Graph.tsv',
                     sep="\t", header=None, names=['V0', 'V1', 'Edge'], skiprows=1)
    df_th = pd.read_csv(path_save + 'Graph_threshold.tsv',
                     sep="\t", header=None, names=['V0', 'V1'], skiprows=1)
    df = pd.merge(df_th, df, on=['V0', 'V1'], how='left')
    for id, paraclique in enumerate(list_parac):
        df_parac = df[(df['V0'].isin(paraclique)) & (df['V1'].isin(paraclique))]

        # create a graph object from the DataFrame
        G = nx.from_pandas_edgelist(df_parac, source='V0', target='V1', edge_attr='Edge')
        # set the position of the nodes using a spring layout
        pos = nx.spring_layout(G)
        # draw the nodes
        nx.draw_networkx_nodes(G, pos, node_color='red', node_size=600)
        # draw the edges
        nx.draw_networkx_edges(G, pos, edge_color='blue', width=1.6)
        # add labels to the edges
        edge_labels = nx.get_edge_attributes(G, 'Edge')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='black',
                                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), label_pos=0.3)

        # add labels to the nodes
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

        # set the axis limits and remove the axis
        plt.xlim([-1.1, 1.1])
        # plt.ylim([-1.1, 1.1])
        plt.axis('off')

        plt.savefig(path_paraclique + f'/paraclique_{id}.png', dpi=300)

        # show the plot
        plt.show()
    return

def CheckAnomalyDistribution():
    '''
    Check anomaly distribution between each stations in paraclique
    :return:
    '''
    with open(path_graph + "/Summary_Anomalies.pickle", "rb") as f:
        # load the dictionary from the file using pickle.load
        dict_pred = pickle.load(f)

    list_parac = []
    with open(path_save + "paraclique.txt", "r") as f:
        for line in f:
            paraclique = line.strip().split('\t')
            list_parac.append(paraclique)

    df = pd.read_csv(path_save + 'Graph.tsv',
                     sep="\t", header=None, names=['V0', 'V1', 'Edge'], skiprows=1)
    df_th = pd.read_csv(path_save + 'Graph_threshold.tsv',
                        sep="\t", header=None, names=['V0', 'V1'], skiprows=1)
    df = pd.merge(df_th, df, on=['V0', 'V1'], how='left')

    dict_edge = {}
    '''
    key:id,
    value: {
        key: edge
        value:{
                key: signal
                value: [timestamp]
                }
            }
        
    '''
    for id, paraclique in enumerate(list_parac):
        df_parac = df[(df['V0'].isin(paraclique)) & (df['V1'].isin(paraclique))]
        dict_edge[id] = {}
        list_edges = []
        for ix, row in df_parac.iterrows():
            v0 = str(row['V0'])
            v1 = str(row['V1'])
            list_edges.append(sorted([v0, v1]))
        for year in dict_pred:
            for date in tqdm.tqdm(dict_pred[year], total=len(dict_pred[year]), desc=f"Paraclique {id} Year {year}:"):
                for signal in dict_pred[year][date]:
                    sta3ns = dict_pred[year][date][signal]
                    for (v0, v1) in list_edges:
                        if v0 in sta3ns and v1 in sta3ns:
                            edge = tuple(sorted([v0, v1]))
                            if edge not in dict_edge[id]:
                                dict_edge[id][edge] = {}
                            if signal not in dict_edge[id][edge]:
                                dict_edge[id][edge][signal] = []
                            dict_edge[id][edge][signal].append(date)
        for edge, v_edge in dict_edge[id].items():
            for signal, dates in v_edge.items():
                v_edge[signal] = sorted(dates, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))

    # open a file for binary writing
    os.makedirs(path_save + "paraclique/", exist_ok=True)
    with open(path_save + "paraclique/dict_paraclique.pickle", "wb") as f:
        # write the dictionary to the file using pickle.dump
        pickle.dump(dict_edge, f)
    # Convert the dictionary to a list of dictionaries
    for id in dict_edge:
        df_anomaly = pd.DataFrame.from_dict(dict_edge[id], orient='columns')
        df_count = df_anomaly.applymap(lambda x: len(x) if isinstance(x, list) else 0)
        df_count['sum'] = df_count.sum(axis=1)
        df_count = df_count.sort_values(by="sum", ascending=False)
        df_count = df_count.transpose()
        df_count.to_csv(path_save + f"paraclique/para{id}_count.csv")

        df_detail = pd.DataFrame.from_dict(dict_edge[id], orient='index').reset_index()
        df_detail['edge'] = list(zip(df_detail['level_0'], df_detail['level_1']))
        # set the index to the new column 'C'
        df_detail.set_index('edge', inplace=True)
        df_detail.drop(['level_0', 'level_1'], axis=1, inplace=True)
        df_detail = df_detail[df_count.columns]
        df_detail.to_csv(path_save + f"paraclique/para{id}_detail.csv")

    return

if __name__ == '__main__':
    domain = 'Con'
    dir_save = 'graph_0322'
    path_results = '../results/{}_allsta3n/'.format(domain)
    path_graph = path_results + dir_save
    # target_year = '2019' # 'all' for all years

    ReadAnomalies() # read all anomalies results from voting output file

    for target_year in ['all', '2019', '2020', '2021'][:1]:
        path_save = os.path.join(path_graph, f'graph_{target_year}/')
        os.makedirs(path_save, exist_ok=True)

        ConvertResults2Graph()
        HistPlot(path_save)
        ScaledWeightGraph(path_save)
        # # use hydra server to find threshold

        FindThreshold_spectral()
        # # use paraclique.sh in the file
        # subprocess.run(['sh', 'paraclique.sh', f'{dir_save}/graph_{target_year}'])
        #
        PlotParaclique()
        CheckAnomalyDistribution()

    print('test finished!!!')
