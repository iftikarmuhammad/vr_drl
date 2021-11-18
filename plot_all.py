import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

log_path = os.getcwd() + "/result/211117_2235/"

# local = pd.read_csv(log_path + "MovingAverageReward_lc_energy.csv")

# mcnoma = pd.read_csv(log_path + "MovingAverageRewardMCNOMA_energy.csv")

# Exh = pd.read_csv(log_path + "MovingAverageReward_exh_energy.csv")

# mc_noma2 = pd.read_csv(log_path + "MovingAverageRewardDQN_energy.csv")

# rand = pd.read_csv(log_path + "MovingAverageReward_rand_energy.csv")

dqn = pd.read_csv(log_path + "MovingAverageEnergyDQN.csv")

# dqn = dqn.div(90)

def plot_gain(dqn, rolling_interval=3, frame=3):

    mpl.style.use('fivethirtyeight')
    csfont = {'fontname':'Times New Roman'}
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor('w')
    fig.set_facecolor('w')
    ax.grid(True)

    # plt.plot(np.arange(frame) + 1, local.rolling(rolling_interval, min_periods=1).mean(), 'b', marker='+', ms=10,
    #          linewidth=3, markevery=1, label='Local')
    # plt.fill_between(np.arange(frame) + 1, local.rolling(rolling_interval, min_periods=1).min(),
    #              local.rolling(rolling_interval, min_periods=1).max()[0], color='b', alpha=0.01)
    
    # plt.plot(np.arange(frame) + 1, rand.rolling(rolling_interval, min_periods=1).mean(), 'r', marker='*', ms=10,
    #          linewidth=3, markevery=1, label='Random')
    # plt.fill_between(np.arange(frame) + 1, rand.rolling(rolling_interval, min_periods=1).min()[0],
    #                  rand.rolling(rolling_interval, min_periods=1).max()[0], color='r', alpha=0.01)

    # plt.plot(np.arange(frame) + 1, mcnoma.rolling(rolling_interval, min_periods=1).mean(), 'g', marker='o', ms=10,
    #          linewidth=3, markevery=1, label='DNN')
    # plt.fill_between(np.arange(frame) + 1, mcnoma.rolling(rolling_interval, min_periods=1).min()[0],
    #                  mcnoma.rolling(rolling_interval, min_periods=1).max()[0], color='g', alpha=0.01)
    
    # plt.plot(np.arange(frame) + 1, Exh.rolling(rolling_interval, min_periods=1).mean(), 'm', marker='<', ms=10,
    #          linewidth=3, markevery=1, label='Exhaustive')
    # plt.fill_between(np.arange(frame) + 1, Exh.rolling(rolling_interval, min_periods=1).min()[0],
    #                  Exh.rolling(rolling_interval, min_periods=1).max()[0], color='m', alpha=0.01)

    plt.plot(np.arange(frame) + 1, dqn.rolling(rolling_interval, min_periods=1).mean(), 'C0', marker='s', ms=10,
             linewidth=3, markevery=1, label='DQN')
    # plt.fill_between(np.arange(frame) + 1,dqn.rolling(rolling_interval, min_periods=1).min()[0],
    #                  dqn.rolling(rolling_interval, min_periods=1).max()[0], color='c', alpha=0.01)

    # plt.plot(np.arange(frame) + 1, mcnoma_2.rolling(rolling_interval, min_periods=1).mean(), 'y',marker='D', ms=10,
    #          linewidth=3, markevery=1, label='DQN Greedy')
    # plt.fill_between(np.arange(frame) + 1, mcnoma_2.rolling(rolling_interval, min_periods=1).min()[0],
    #                  mcnoma_2.rolling(rolling_interval, min_periods=1).max()[0], color='y', alpha=0.01)

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc='upper center', shadow=True, fontsize=15,
               facecolor=fig.get_facecolor())

    plt.ylabel('Reward', **csfont, fontsize=22)
    plt.xlabel('Number of RBs', **csfont, fontsize=22)
    plt.savefig('MultiChannelMovingAverageRate.png', facecolor=fig.get_facecolor())
    plt.show()

plot_gain(dqn['dqn'])

# plot_gain(local['local'], mcnoma['mcnoma'], Exh['exhaustive'],  Edge['edge'], droo['DROO'])

