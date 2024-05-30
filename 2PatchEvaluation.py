'''
Evaluate the patch clusters in our method
'''
import math

import pandas as pd
import os
import re
import pickle
import sys
import tqdm
import subprocess
import numpy as np
from datetime import datetime
import networkx as nx
import glob
from matplotlib import pyplot as plt

def HistPlot(path_save):
    num_bin = 20
    df = pd.read_csv(path_save + 'Graph.tsv',
                     sep="\t", header=None, names=['V0', 'V1', 'Edge'], skiprows=1)
    df['Edge_scale'] = df['Edge']/df['Edge'].max()
    fig = plt.figure(figsize=(6.5, 5))
    plt.hist(df['Edge_scale'], bins=num_bin, range=(0, 1), alpha=0.8, edgecolor='black', linewidth=1.0)
    # plt.yscale('log')
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
            value = float(anomaly.split(':')[1])
            if '_normalized' in path:
                signal += '-normalized'
            if signal not in dict_pred[date]:
                dict_pred[date][signal] = []
            dict_pred[date][signal].append((sta3n, value))

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


def PatchResults2Graph():
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


    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    dict_pred_date = {}

    year = '2020'
    for date in tqdm.tqdm(dict_pred[year], total=len(dict_pred[year]), desc=f"Year {year}:"):
        date_t = datetime.strptime(date, '%Y-%m-%d').date()
        if start_date <= date_t <= end_date:
            dict_pred_date[date] = dict_pred[year][date]

    for date in dict_pred_date:
        # signals = dict_pred_date[date]
        for signal in dict_pred_date[date]:
            list_sta3n = list(set([results[0] for results in dict_pred_date[date][signal]]))
            dict_value = {}
            for item in dict_pred_date[date][signal]:
                (sta,value) = item
                if sta not in dict_value:
                    dict_value[sta] = value
            if len(list_sta3n) > 1:  # more than 1 stations
                for i in range(len(list_sta3n) - 1):
                    for j in range(i + 1, len(list_sta3n)):
                        edge = frozenset([list_sta3n[i], list_sta3n[j]])
                        if len(edge) != 2:
                            print(f'false in signal {signal} in date {date}')
                            sys.exit(1)
                        if edge not in dict_graph:
                            dict_graph[edge] = 0
                        v_i = dict_value[list_sta3n[i]]
                        v_j = dict_value[list_sta3n[j]]
                        w = math.log2(v_i/2+1) * math.log2(v_j/2+1)
                        dict_graph[edge] += w

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


def PatchResults2GraphWindow():
    '''
    Add window to recognize shift of anomalies. As long as two anomalies are within a time range, we consider them as the same anomalies.
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


    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    dict_pred_date = {}

    year = '2020'
    for date in tqdm.tqdm(dict_pred[year], total=len(dict_pred[year]), desc=f"Year {year}:"):
        date_t = datetime.strptime(date, '%Y-%m-%d').date()
        if start_date <= date_t <= end_date:
            dict_pred_date[date] = dict_pred[year][date]
    list_date = list(dict_pred_date.keys())
    list_date = sorted(list_date, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    list_checked_date = []

    for i, date_i in enumerate(list_date[:-1]):
        for j, date_j in enumerate(list_date[i+1:]):
            dtime_i = datetime.strptime(date_i, '%Y-%m-%d').date()
            dtime_j = datetime.strptime(date_j, '%Y-%m-%d').date()
            if (dtime_j-dtime_i).days <= window: # if less than window size
                list_checked_date.append(date_i)
                list_checked_date.append(date_j)
                signal_i = list(dict_pred_date[date_i].keys())
                signal_j = list(dict_pred_date[date_j].keys())
                signals = list(set(signal_i+signal_j))
                for signal in signals:
                    list_sta3n = []
                    if signal in signal_i:
                        list_sta3n += dict_pred_date[date_i][signal]
                    if signal in signal_j:
                        list_sta3n += dict_pred_date[date_j][signal]
                    if len(list_sta3n) > 1:  # more than 1 stations
                        for m in range(len(list_sta3n) - 1):
                            for n in range(m + 1, len(list_sta3n)):
                                if list_sta3n[m][0] != list_sta3n[n][0]:
                                    edge = frozenset([list_sta3n[m][0], list_sta3n[n][0]])
                                    if len(edge) != 2:
                                        print(f'false in signal {signal} in date {date_i} and {date_j}')
                                        sys.exit(1)
                                    if edge not in dict_graph:
                                        dict_graph[edge] = 0
                                    v_m = list_sta3n[m][1]
                                    v_n = list_sta3n[n][1]
                                    w = math.log2(v_m*k + 1) * math.log2(v_n*k + 1)
                                    dict_graph[edge] += w
            else:
                # print()
                if date_i not in list_checked_date:
                    list_checked_date.append(date_i)
                    dict_graph = BuildGraph4Date(date_i, dict_graph, dict_pred_date)

                if (date_j == list_date[-1]) and (date_j not in list_checked_date):
                    list_checked_date.append(date_j)
                    dict_graph = BuildGraph4Date(date_j, dict_graph, dict_pred_date)
                break

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

def BuildGraph4Date(date, dict_graph, dict_pred_date):
    for signal in dict_pred_date[date]:
        list_sta3n = list(set([results[0] for results in dict_pred_date[date][signal]]))
        dict_value = {}
        for item in dict_pred_date[date][signal]:
            (sta,value) = item
            if sta not in dict_value:
                dict_value[sta] = value
        if len(list_sta3n) > 1:  # more than 1 stations
            for i in range(len(list_sta3n) - 1):
                for j in range(i + 1, len(list_sta3n)):
                    edge = frozenset([list_sta3n[i], list_sta3n[j]])
                    if len(edge) != 2:
                        print(f'false in signal {signal} in date {date}')
                        sys.exit(1)
                    if edge not in dict_graph:
                        dict_graph[edge] = 0
                    v_i = dict_value[list_sta3n[i]]
                    v_j = dict_value[list_sta3n[j]]
                    w = math.log2(v_i*k+1) * math.log2(v_j*k+1)
                    dict_graph[edge] += w
    return dict_graph

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
    threshold = df.loc[df['2nd-eigenvalue'].idxmin()]['threshold']
    # min_eigen = df['2nd-eigenvalue'].min()
    # threshold = 0.6
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

def PlotGraph():
    df_th = pd.read_csv(path_save + 'Graph_threshold.tsv',
                        sep="\t", header=None, names=['V0', 'V1'], skiprows=1)

    # Create a graph object from the DataFrame
    G = nx.from_pandas_edgelist(df_th, source='V0', target='V1')

    # Set the position of the nodes using a spring layout
    layout = nx.spring_layout(G)

    # Create a figure and axes objects
    fig = plt.figure(figsize=(10, 8))

    # Draw the nodes
    nx.draw_shell(G, with_labels=True, node_color='lightblue')
    # nx.draw_networkx_edges(G, layout, edge_color='#AAAAAA')
    # plt.tight_layout()

    plt.savefig(path_save + '/graph_th.png', dpi=300)

    # show the plot
    plt.show()
    return

def PlotGraph_v2():
    df_th = pd.read_csv(path_save + 'Graph_threshold.tsv',
                        sep="\t", header=None, names=['V0', 'V1'], skiprows=1)

    list_node = list(set(df_th.V0.to_list() + df_th.V1.to_list()))

    fig = plt.figure(figsize=(10, 10))

    # Create a graph object from the DataFrame
    g = nx.from_pandas_edgelist(df_th, source='V0', target='V1')

    # Set the position of the nodes using a spring layout
    layout = nx.kamada_kawai_layout(g, scale=1)

    # 3. Draw the parts we want
    nx.draw_networkx_edges(g, layout, edge_color='#AAAAAA')

    nx.draw_networkx_nodes(g, layout, node_size=100, node_color='#AAAAAA')

    high_degree_people = [node for node in g.nodes() if node in list_node and g.degree(node) >= 6]
    nx.draw_networkx_nodes(g, layout, nodelist=high_degree_people, node_size=100, node_color='#fc8d62')

    # club_dict = dict(zip(high_degree_people, high_degree_people))
    nx.draw_networkx_labels(g, layout, font_size=8)
    plt.savefig(path_save + '/graph_th_2.png', dpi=300)

    # show the plot
    plt.show()
    return

def PlotParaclique():
    path_paraplot = path_paraclique + 'plot/'
    os.makedirs(path_paraplot, exist_ok=True)

    list_parac = []
    with open(path_paraclique + "paraclique.txt", "r") as f:
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
        layout = nx.kamada_kawai_layout(G)
        # draw the nodes
        nx.draw_networkx_edges(G, layout, edge_color='#AAAAAA')

        nx.draw_networkx_nodes(G, layout, node_size=300, node_color='#fc8d62')

        # # add labels to the edges
        # edge_labels = nx.get_edge_attributes(G, 'Edge')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='black',
        #                              bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), label_pos=0.3)

        # add labels to the nodes
        # nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
        nx.draw_networkx_labels(G, layout, font_size=8)

        # set the axis limits and remove the axis
        # plt.xlim([-1.1, 1.1])
        # plt.ylim([-1.1, 1.1])
        plt.axis('off')

        plt.savefig(path_paraplot + f'/paraclique_{id}.png', dpi=300)

        # show the plot
        plt.show()
    return

def Recall():
    path = '../data/system_patch/patch_count_threshold.csv'
    df = pd.read_csv(path, index_col=False)

    df['sta3ns'] = df.apply(lambda row: re.findall(r'\((\d+)\)', row['facility']), axis=1)
    clusters = df['sta3ns'].to_list()

    list_stations = list(set([item for sublist in clusters for item in sublist]))

    TP = []
    clusters_detected = []
    with open(path_paraclique + "paraclique.txt", "r") as f:
        for line in f:
            paraclique = line.strip().split('\t')
            clusters_detected.append(paraclique)
            for cluster in clusters:
                detected = list(set(cluster) & set(paraclique))
                if len(detected) > 1:
                    TP.append(detected)
    num_TP = sum([len(item) for item in TP])
    recall = num_TP/len(list_stations)
    print(f'Recall: {recall}')
    df_recall = pd.DataFrame.from_dict({'window_size':[window], 'glom': [glom], 'recall': [recall]})
    df_recall.to_csv(path_paraclique+'recall.csv', index=False)
    return

def ReadRecall():
    df_recall = pd.DataFrame(columns=['Lookback_period', 'Max_defer', 'glom', 'Recall'])
    for m in range(1, 5):
        for window in range(0,6):
            dir_save = f'graph_0408_{m}week'
            path_graph = path_results + dir_save
            path_save = os.path.join(path_graph, f'graph_patch_window{window}/')
            for glom in [1,2]:
                path_paraclique = path_save + f'paraclique_glom{glom}/'
                recall = pd.read_csv(path_paraclique+'recall.csv', index_col=False)
                r = recall.at[0,'recall']
                new_row = {'Lookback_period': m, 'Max_defer': window, 'glom': glom, 'Recall': r}
                df_recall = df_recall.append(new_row, ignore_index=True)
    df_recall.to_csv(path_results + 'patch_evaluation.csv', index=False)
    return

def PlotResults():
    df = pd.read_csv(path_results + 'patch_evaluation.csv', index_col=False)
    path_plot = path_results + 'plot/'
    os.makedirs(path_plot, exist_ok=True)
    # colors = ['turquoise', 'salmon', 'teal', 'purple', 'crimson', 'dodgerblue', 'gold', 'limegreen']
    markers = ['v', '^', 'd', 'o', '<', '>', 'X',  '*']
    # colors = ['darkslategray', 'tomato', 'teal', 'purple', 'crimson', 'royalblue', 'forestgreen', 'sienna']
    colors = ['purple', 'orangered', 'royalblue', 'forestgreen']

    list_lbp = df['Lookback_period'].unique().tolist()
    list_md = df['Max_defer'].unique().tolist()
    list_g = df['glom'].unique().tolist()
    x = list_md

    fig = plt.figure(figsize=(10, 7))
    plt.xlabel('Max Defer Time (days)', fontsize=16)
    plt.ylabel('Detection Rate ' + ' (%)', fontsize=16)
    for i, lbp in enumerate(list_lbp):
        for j, g in enumerate(list_g):
            y = df[(df['Lookback_period'] == lbp) & (df['glom'] == g)]['Recall'].tolist()
            y = [r*100 for r in y]
            plt.plot(x,y, colors[i], marker=markers[len(list_g)*i+j], markersize=9, linewidth=1.6,
                     linestyle='dashed' if j==1 else 'solid', label=f'Lookback={int(lbp)}, glom={int(g)}', alpha=0.85)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=11)
    # plt.ylim(, 105)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    fig.savefig(path_plot + 'Patch_Eva.png', dpi=600)
    plt.show()
    plt.close()
    return

# def PlotBaseline(df, methods, acc, xlabel, ylabel, savefile):
#     path_plot = path_test + 'plot/'
#     os.makedirs(path_plot, exist_ok=True)
#     colors = ['turquoise', 'salmon' ,'teal', 'purple', 'crimson','dodgerblue']
#
#     fig = plt.figure(figsize=(6, 5))
#     plt.xlabel(xlabel, fontsize=16)
#     plt.ylabel(ylabel + ' (%)', fontsize=16)
#     # y = []
#     for i, m in enumerate(methods):
#         y = df[df['method'] == m][acc].tolist()
#         if m in ['PCA', 'LogCluster']:
#             y = [r*100 for r in y]
#         x = list(range(len(y)))
#         plt.plot(x,y, colors[i], marker='.', markersize=14, linewidth=2.0, label=m, alpha=0.9)
#     # plt.legend(title='Window Size',title_fontsize=14)
#     plt.legend(bbox_to_anchor=(0.5, 1.23),loc='upper center', ncol=3, fontsize=12)
#     plt.ylim(-2,105)
#     plt.xticks(np.arange(0, 5, 1), [10, 20, 50, 100, 200], fontsize=14)
#     plt.yticks(np.arange(0, 110, 20), fontsize=14)
#     plt.tight_layout()
#     plt.show()
#     fig.savefig(path_plot + savefile, dpi=600)
#     plt.close()

if __name__ == '__main__':
    domain = 'Con'
    start_date_str = '2020-7-1'
    end_date_str = '2020-8-31'
    window = 5 # days
    k = 1/4 # coeffient for value of anomalies
    glom = 1
    path_results = '../results/{}_allsta3n_patch/'.format(domain)
    for m in [1,3,4]:
        for window in range(0,6):
            dir_save = f'graph_0408_{m}week'
    
            path_graph = path_results + dir_save
    
            path_save = os.path.join(path_graph, f'graph_patch_window{window}/')
            os.makedirs(path_save, exist_ok=True)
    
            # path_paraclique = path_save + f'paraclique_glom{glom}/'
            # os.makedirs(path_paraclique, exist_ok=True)
    
            ReadAnomalies() # read all anomalies results from voting output file
    
    
            # PatchResults2Graph()
            PatchResults2GraphWindow()
            HistPlot(path_save)
            # ScaledWeightGraph(path_save)
            # # # # use hydra server to find threshold
            # #
            FindThreshold_spectral()
            # PlotGraph_v2()
    
            ## use paraclique.sh in the file
            for glom in [1,2]:
                path_paraclique = path_save + f'paraclique_glom{glom}/'
                os.makedirs(path_paraclique, exist_ok=True)
                subprocess.run(['sh', 'paraclique.sh', path_save, path_paraclique, str(glom)])
                # #
                PlotParaclique()
                Recall()

    ReadResults
    ReadRecall()
    PlotResults()
