import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import os




def PlotResults1Fig(raw_data, x_labels, pred_high, pred_mid, pred_low, path_save_file):
    # plot raw data into one figure
    fig = plt.figure(figsize=(8, 5))
    x = [i + 1 for i in range(len(raw_data))]
    plt.plot(x, raw_data, 'grey', label='data')
    if len(raw_data) > 15:
        interval = round(len(raw_data) / 15)
        x_x = np.arange(min(x), max(x) + interval - 1, interval)
    else:
        x_x = np.arange(min(x), max(x), 1)
    x_x_labels = [x_labels[id - 1] for id in x_x]
    plt.xticks(x_x, x_x_labels, rotation='90')
    flag_first_bkp = False
    for idx, bkp in enumerate(pred_high):
        if (bkp == 1):
            if (flag_first_bkp == False):
                plt.scatter(idx + 1, raw_data[idx], c='r', s=80, label='High-confidence detection')
                flag_first_bkp = True
            else:
                plt.scatter(idx + 1, raw_data[idx], c='r', s=80)
    # subplot 2
    flag_first_bkp = False
    for idx, bkp in enumerate(pred_mid):
        if (bkp == 1):
            if (flag_first_bkp == False):
                plt.scatter(idx + 1, raw_data[idx], c='orange', s=80, label='Mid-confidence detection')
                flag_first_bkp = True
            else:
                plt.scatter(idx + 1, raw_data[idx], c='orange', s=80)
    # subplot 3
    flag_first_bkp = False
    for idx, bkp in enumerate(pred_low):
        if (bkp == 1):
            if (flag_first_bkp == False):
                plt.scatter(idx + 1, raw_data[idx], c='g', s=80, label='Low-confidence detection', zorder=2)
                flag_first_bkp = True
            else:
                plt.scatter(idx + 1, raw_data[idx], c='g', s=80)
    # plt.title(path_save.split('/')[-2] + '_LowProb')

    fig.tight_layout()
    plt.legend()
    plt.show()
    fig.savefig(path_save_file, dpi=300)
    plt.close()

def PlotSplitFig(raw_data, x_labels, pred_high, pred_mid, pred_low, path_save_file):
    # plot results
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.set_size_inches(8, 5)
    # ax1 = plt.subplot(3, 1, 1)
    x = [i + 1 for i in range(len(raw_data))]
    axs[0].plot(x, raw_data, 'darkgrey')
    x_x = np.arange(min(x), max(x) + 1, 100)
    x_x_labels = [' ' for id in x_x]
    plt.xticks(x_x, x_x_labels, rotation='90')
    flag_first_bkp = False
    for idx, bkp in enumerate(pred_high):
        if (bkp == 1):
            if (flag_first_bkp == False):
                axs[0].scatter(idx + 1, raw_data[idx], c='r', s=80, label='anomaly')
                flag_first_bkp = True
            else:
                axs[0].scatter(idx + 1, raw_data[idx], c='r', s=80)
    axs[0].title.set_text('High_confidence')
    # subplot 2
    x = [i + 1 for i in range(len(raw_data))]
    # plot raw data
    axs[1].plot(x, raw_data, 'darkgrey')
    x_x = np.arange(min(x), max(x) + 1, 100)
    x_x_labels = [' ' for id in x_x]
    plt.xticks(x_x, x_x_labels, rotation='90')
    axs[1].title.set_text('Mid_confidence')
    flag_first_bkp = False
    for idx, bkp in enumerate(pred_mid):
        if (bkp == 1):
            if (flag_first_bkp == False):
                axs[1].scatter(idx + 1, raw_data[idx], c='orange', s=80, label='anomaly')
                flag_first_bkp = True
            else:
                axs[1].scatter(idx + 1, raw_data[idx], c='orange', s=80)
    # subplot 3
    x = [i + 1 for i in range(len(raw_data))]
    # plot raw data
    axs[2] = plt.plot(x, raw_data, 'darkgrey')
    x_x = np.arange(min(x), max(x) + 1, 100)
    x_x_labels = [x_labels[id - 1] for id in x_x]
    plt.xticks(x_x, x_x_labels, rotation='90')
    flag_first_bkp = False
    for idx, bkp in enumerate(pred_low):
        if (bkp == 1):
            if (flag_first_bkp == False):
                plt.scatter(idx + 1, raw_data[idx], c='g', s=80, label='anomaly')
                flag_first_bkp = True
            else:
                plt.scatter(idx + 1, raw_data[idx], c='g', s=80)
    plt.title('LowProb_confidence')

    fig.tight_layout()

    fig.savefig(path_save_file)
    plt.close()

def Plot7Preds(df, path_save):
    x_labels = df['timestamp'].tolist()
    # get the methods number of results
    pred = list(df)[2:]
    fig, ax = plt.subplots(nrows=len(pred), ncols=1, figsize=(10,10))
    x = [i + 1 for i in range(len(df))]
    if len(x) > 15:
        interval = round(len(x) / 15)
        # x_x = np.arange(min(x), max(x) + interval - 1, interval)
        x_x = np.arange(min(x), max(x), interval)

    else:
        x_x = np.arange(min(x), max(x), 1)
    x_x_labels = [x_labels[id - 1] for id in x_x]
    flag_3 = False
    flag_2 = False
    flag_1 = False
    for i, p in enumerate(pred):
        ax[i].plot(x, df['value'])  # row=0, col=0
        ax[i].set_ylabel(p)
        ax[i].set_xticklabels([])
        for idx, bkp in enumerate(df[p]):
            if bkp == 3:
                if not flag_3:
                    ax[i].scatter(idx + 1, df['value'][idx], c='r', s=50, label = 'High-confidence anomaly')
                    flag_3 = True
                else:
                    ax[i].scatter(idx + 1, df['value'][idx], c='r', s=50)
            elif bkp == 2:
                if not flag_2:
                    ax[i].scatter(idx + 1, df['value'][idx], c='orange', s=50, label = 'Mid-confidence anomaly')
                    flag_2 = True
                else:
                    ax[i].scatter(idx + 1, df['value'][idx], c='orange', s=50)
            elif bkp == 1:
                if not flag_1:
                    ax[i].scatter(idx + 1, df['value'][idx], c='green', s=50, label = 'Low-confidence anomaly')
                    flag_1 = True
                else:
                    ax[i].scatter(idx + 1, df['value'][idx], c='green', s=50)
    plt.xticks(x_x, x_x_labels, rotation='90')
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(0.87, 1.012),
          ncol=3, fancybox=True)
    plt.tight_layout()
    plt.show()
    # img = io.BytesIO()
    # FigureCanvas(fig).print_png(img)
    # plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    # return plot_url
    fig.savefig(path_save + 'results.png', dpi=300)
    plt.close()

def Plot7PredsUpperBound(df, path_save):
    x_labels = df['timestamp'].tolist()
    # get the methods number of results
    pred = list(df)[2:]
    fig, ax = plt.subplots(nrows=len(pred), ncols=1, figsize=(10,10))
    x = [i + 1 for i in range(len(df))]
    if len(x) > 15:
        interval = round(len(x) / 15)
        # x_x = np.arange(min(x), max(x) + interval - 1, interval)
        x_x = np.arange(min(x), max(x), interval)

    else:
        x_x = np.arange(min(x), max(x), 1)
    x_x_labels = [x_labels[id - 1] for id in x_x]
    for i, p in enumerate(pred):
        ax[i].plot(x, df['value'])  # row=0, col=0
        ax[i].set_ylabel(p)
        ax[i].set_xticklabels([])
        for idx, bkp in enumerate(df[p]):
            if bkp == 1:
                ax[i].scatter(idx + 1, df['value'][idx], c='red', s=50)
    plt.xticks(x_x, x_x_labels, rotation='90')
    plt.tight_layout()
    # plt.show()
    fig.savefig(path_save + 'results.png', dpi=300)
    plt.close()

def PlotCPD(df, path_save):
    x_labels = df['timestamp'].tolist()
    # get the methods number of results
    pred = list(df)[2:]
    fig, ax = plt.subplots(nrows=len(pred), ncols=1, figsize=(10, 10))
    x = [i + 1 for i in range(len(df))]
    if len(x) > 15:
        interval = round(len(x) / 15)
        # x_x = np.arange(min(x), max(x) + interval - 1, interval)
        x_x = np.arange(min(x), max(x), interval)

    else:
        x_x = np.arange(min(x), max(x), 1)
    x_x_labels = [x_labels[id - 1] for id in x_x]
    for i, p in enumerate(pred):
        ax[i].plot(x, df['value'])  # row=0, col=0
        ax[i].set_ylabel(p)
        ax[i].set_xticklabels([])
        for idx, bkp in enumerate(df[p]):
            if bkp == 1:
                ax[i].scatter(idx + 1, df['value'][idx], c='red', s=50)
    plt.xticks(x_x, x_x_labels, rotation='90')
    plt.title('Changpoint probability (offline + online)')
    plt.tight_layout()
    plt.show()
    fig.savefig(path_save + 'CPD.png', dpi=300)
    plt.close()

def PlotResultsAnomaly(df, path_save, anomaly_range):
    x_labels = df['timestamp'].tolist()
    # get the methods number of results
    pred = list(df)[2:]
    fig, ax = plt.subplots(nrows=len(pred), ncols=1, figsize=(10,10))
    x = [i + 1 for i in range(len(df))]
    if len(x) > 15:
        interval = round(len(x) / 15)
        # x_x = np.arange(min(x), max(x) + interval - 1, interval)
        x_x = np.arange(min(x), max(x), interval)

    else:
        x_x = np.arange(min(x), max(x), 1)
    x_x_labels = [x_labels[id - 1] for id in x_x]
    for i, p in enumerate(pred):
        ax[i].plot(x, df['value'])  # row=0, col=0
        ax[i].set_ylabel(p)
        ax[i].set_xticklabels([])
        for idx, bkp in enumerate(df[p]):
            if bkp == 1:
                ax[i].scatter(idx + 1, df['value'][idx], c='red', s=50)
        ax[i].vlines(x=anomaly_range[0]+1, ymin=np.min(df['value']), ymax=np.max(df['value']), colors='green', ls='dashed', lw=2,
                      label='Injected data range')
        ax[i].vlines(x=anomaly_range[1]+1, ymin=np.min(df['value']), ymax=np.max(df['value']), colors='green', ls='dashed', lw=2,
                      label='Injected data range')
    plt.xticks(x_x, x_x_labels, rotation='90')
    plt.tight_layout()
    # plt.show()
    fig.savefig(path_save + '.png', dpi=300)
    plt.close()
