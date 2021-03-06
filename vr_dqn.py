import scipy.io as sio
import numpy as np
import random as rn
import time as tm
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

from dqn import DeepQNetwork
from vr_env import step


def ma(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def flatten(actions) :
    # This function flattens any actions passed somewhat like so -:
    # INPUT -: [[1, 2, 3], 4, 5]
    # OUTPUT -: [1, 2, 3, 4, 5]
    
    new_actions = [] # Initializing the new flattened list of actions.
    for action in actions :
        # print(action)
        # Loop through the actions
        if type(action) == list :
            # If any actions is a pair of actions i.e. a list e.g. [1, 1] then
            # add it's elements to the new_actions list.
            new_actions += action
        elif type(action) == int :
            # If the action is an integer then append it directly to the new_actions
            # list.
            new_actions.append(action)
    return new_actions

def generate_actions(possible_actions):
    if len(possible_actions) == 1 :
        return possible_actions
    pairs = [] # Initializing a list to contain all pairs of actions generated.
    for act in possible_actions[0] :
        for act2 in possible_actions[1] :
            pairs.append(flatten([act, act2]))
    new_possible_actions = [pairs] + possible_actions[2 : ]
    possible_action_vectors = generate_actions(new_possible_actions)
    return possible_action_vectors

def run(eps, t, s, ue, plot, result_dir):

    # This algorithm generates K modes from DNN, and chooses with largest
    # reward. The mode with largest reward is stored in the memory, which is
    # further used to train the DNN.
    print("DQN Computing Phase")
    nu = ue  # number of users
    n = t  # number of time frames
    b = 1 * 10 ** 6
    SUB_NUM = s
    PLAYOUT_DELAY = 0.1

    NUM_EPISODES = eps
    INITIAL_EPSILON = 0.4
    FINAL_EPSILON = 0.01
    EPSILON_DECAY = 3*n*NUM_EPISODES/4
    TRAINING_CYCLE = 10
    TARGET_UPDATE_CYCLE = 1
    epsilon = INITIAL_EPSILON

    w1 = 1
    # w2 = 1
    w2 = 100

    RESULT_DIR = datetime.now().strftime("%y%m%d_%H%M")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    k = 5    # overfill constant

    # Load data
    channel = sio.loadmat('data/ch_gain/data_%d' % 10)['input_h']
    channel = channel * 1000000

    v_df = pd.read_csv('data/ang_vec/ang_vec_3users.csv')
    vstd_df = v_df                                          # standardized/normalized angular velocity
    v_df = v_df.clip(-4,4)

    for i in range(nu):
        user = 'user%s'%str(i+1)
        scaler = StandardScaler()
        median = vstd_df.loc[(vstd_df[user]<4) & vstd_df[user] >-4, user].median()
        vstd_df[user] = vstd_df[user].mask(vstd_df[user]>4, median)
        vstd_df[user] = vstd_df[user].mask(vstd_df[user]<-4, median)
        vstd_df[user] = scaler.fit_transform(vstd_df[user].values.reshape(-1,1))

    a_d = {}

    a_df = pd.read_csv('data/ang/ang_3users.csv')
    a_df = a_df.clip(-2*math.pi, 2*math.pi)

    for i in range(nu):
        user = 'user%s'%str(i+1)
        a_d[user] = a_df[[user+'r', user+'p', user+'y']].mean(axis=1)
        astd_df = pd.DataFrame(a_d)
        scaler = StandardScaler()
        median = astd_df.loc[(astd_df[user]<(2*math.pi)) & astd_df[user] >(-2*math.pi), user].median()
        astd_df[user] = astd_df[user].mask(astd_df[user]>(2*math.pi), median)
        astd_df[user] = astd_df[user].mask(astd_df[user]<(-2*math.pi), median)
        astd_df[user] = scaler.fit_transform(astd_df[user].values.reshape(-1,1))

    # print(len(channel))
    adj_channel = []        #adjust
    adj_users = []

    # Cluster devices into sub-groups (up to a number equivalent to channel index)
    for tens in channel:
        counter = ue
        for single in tens:
            adj_users.append(single) # Add devices to a cluster
            counter = counter - 1
            if counter == 0:
                adj_channel.append(adj_users) # add a cluster to a channel
                adj_users = []
                break
            else:
                continue

    adj_channel = np.asarray(adj_channel)

    sub_range = len(adj_channel)//SUB_NUM # Number of channels per subcarrier
    start = 0
    end = sub_range
    all_sub = []

    for w in range(SUB_NUM):
        sub = adj_channel[start:(end+1)]
        all_sub.append(sub)
        start = end
        end = end+sub_range

    all_subs = [x/1000000 for x in all_sub]    # all_subs used for computation rate calculation
    split_idx = int(len(adj_channel))//SUB_NUM  # training data indices

    ch_df = pd.DataFrame(all_sub[0])

    # print(ch_df)

    for i in range(nu):
        scaler = StandardScaler()
        ch_df[i] = scaler.fit_transform(ch_df[i].values.reshape(-1,1))

    # print(ch_df)

    discrete_action = True

    if discrete_action:
        out_size = pow(SUB_NUM+1, nu) * pow(k, nu)

    network = DeepQNetwork(net=[(nu,4), 120, 80, out_size], # Input layer (nu), 120 and 80 neurons of hidden layers and output layer (nu)
                    learning_rate=0.01,
                    batch_size=64,
                    training_cycle = TRAINING_CYCLE,
                    target_update_cycle = TARGET_UPDATE_CYCLE,
                    sub_num = SUB_NUM,
                    enable_DDQN = False
                    )

    action_space = [SUB_NUM if i%2 else k-1 for i in range(1,2*nu+1)]
    # print(action_space)
    possible_actions = [list(range(0, (k + 1))) for k in action_space]

    # print(possible_actions)
    action_map = generate_actions(possible_actions)[0]

    # print(len(action_map))

    total_rwd = []

    result_dict = defaultdict(list)

    time_duration = []

    delay_idx = int(PLAYOUT_DELAY/0.01)

    for episode in range(NUM_EPISODES):
        episode_reward = 0

        for i in range(n-2):  # Time frames looping
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            start_time = tm.time()

            # i_idx = (i + (episode * n)) % split_idx

            vstd_array = list(vstd_df.iloc[i])
            astd_array = list(astd_df.iloc[i])
            v_array = list(v_df.iloc[i])
            ch_array = list(ch_df.iloc[i])

            result_dict['velocity'].append(np.array(v_array))

            if i >= delay_idx:
                prev_a_array = list(a_df.iloc[i-delay_idx])
                prev_asdt_array = list(astd_df.iloc[i-delay_idx])
            else:
                prev_a_array = list(a_df.iloc[0])
                prev_asdt_array = list(astd_df.iloc[0])

            current_a_array = list(a_df.iloc[i])
            
            state = (ch_array, vstd_array, astd_array, prev_asdt_array)

            state = tf.expand_dims(state,0)

            if np.random.uniform() < epsilon:
                action = rn.randint(0,out_size-1)
            else:
                tmp = network.evaluation_network(state)
                action = tf.argmax(tmp[0])

            action_q = action_map[action]
            env_dict = step(i, action_q, SUB_NUM, ue, b, all_subs, v_array, prev_a_array, current_a_array)
            # print('energy : %s' % sum(env_dict['energy']))
            # print('bb : %s' % (w2  * sum(env_dict['blackborder'])))

            reward = -w1 * sum(env_dict['energy']) - w2 * sum(env_dict['blackborder'])

            vstd_array = list(vstd_df.iloc[i+1])
            ch_array = list(ch_df.iloc[i+1])
            astd_array = list(astd_df.iloc[i+1])

            if i >= delay_idx:
                prev_a_array = list(a_df.iloc[i-delay_idx+1])
                prev_asdt_array = list(astd_df.iloc[i-delay_idx+1])
            else:
                prev_a_array = list(a_df.iloc[0])
                prev_asdt_array = list(astd_df.iloc[0])

            next_state = (ch_array, vstd_array, astd_array, prev_asdt_array)
            next_state = tf.expand_dims(next_state,0)

            network.append_experience({'state':state,
            'action':[action],'reward':[reward],'next_state': next_state})
            episode_reward += reward

            if network.training_counter >= network.training_cycle:
                network.train()
                network.delete_experience()

            duration = tm.time()-start_time
            time_duration.append(duration)

            for k, v in env_dict.items():
                result_dict[k].append(v)
                result_dict['sum_' + k].append(sum(v))

            result_dict['sum_reward'].append(reward)

        print('Episode {} finished after {} timesteps, total rewards {}, epsilon {}'.format(episode, t+1, episode_reward, epsilon))
        total_rwd.append(episode_reward/n)

    eval_dict = defaultdict(list)
    evaluation = True

    if evaluation:
        for i in range(n-2):
            vstd_array = list(vstd_df.iloc[i])
            astd_array = list(astd_df.iloc[i])
            v_array = list(v_df.iloc[i])
            ch_array = list(ch_df.iloc[i])

            result_dict['velocity'].append(np.array(v_array))

            if i >= delay_idx:
                prev_a_array = list(a_df.iloc[i-delay_idx])
                prev_asdt_array = list(astd_df.iloc[i-delay_idx])
            else:
                prev_a_array = list(a_df.iloc[0])
                prev_asdt_array = list(astd_df.iloc[0])
            current_a_array = list(a_df.iloc[i])

            state = (ch_array, vstd_array, astd_array, prev_asdt_array)
            state = tf.expand_dims(state,0)

            tmp = network.evaluation_network(state)
            action = tf.argmax(tmp[0])

            action_q = action_map[action]
            env_dict = step(i, action_q, SUB_NUM, ue, b, all_subs, v_array, prev_a_array, current_a_array)

            reward = -w1 * sum(env_dict['energy']) - w2 * sum(env_dict['blackborder'])

            for k, v in env_dict.items():
                eval_dict[k].append(v)
            eval_dict['reward'].append(reward)

    avg_ue_e = np.mean(result_dict['energy'], axis=0)
    # avg_ue_rwd = np.mean(result_dict['reward'], axis=0)

    avg_e = sum(result_dict['sum_energy'])/(NUM_EPISODES * n)
    avg_rwd = sum(result_dict['sum_reward'])/(NUM_EPISODES * n)

    rwd_df = pd.DataFrame(total_rwd)
    rolling_mean = rwd_df.rolling(5).mean()
    rolling_std = rwd_df.rolling(5).std()
    rwd_df['mean'] = rolling_mean
    rwd_df['std_high'] = rolling_mean + rolling_std
    rwd_df['std_low'] = rolling_mean - rolling_std

    # for i in range(0, len(result_dict['L'][0])):
    #     fig, ax = plt.subplots()
    #     ax.grid(True)
    #     # plt.plot(np.array(result_dict['L'])[:,i], color='b', label='Original')
    #     # plt.plot(np.array(result_dict['optimal_area'])[:,i], color='r', label='Calculated')
    #     plt.plot(np.array(result_dict['velocity'])[-290:,i], color='C%s'%i, label='Velocity')
    #     # plt.plot(np.array(result_dict['velocity'])[:,i], color='r', label='Calculated')
    #     plt.title('Overfill Area User %s' %str(i+1))
    #     plt.xlabel('Timestep')
    #     plt.ylabel('Area (Pixels)')
    #     plt.legend(loc='upper right')
    #     # plt.savefig(result_dir + 'area_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
    #     plt.show()
    #     # plt.close(fig)

    plot_eval = True

    if plot_eval:
        eval_dir = result_dir + 'eval/'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        for i in range(0, len(eval_dict['x'][0])):
            fig, ax = plt.subplots()
            plt.scatter([j for j in range(0,len(eval_dict['x']))], np.array(eval_dict['x'])[:,i], color='C%s'%i)
            plt.title('Subchannel Allocation User %s' %str(i+1))
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Timestep')
            plt.ylabel('Subchannel')
            plt.savefig(eval_dir + 'subchannel_allocation_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(eval_dict['k_coef'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.scatter([j for j in range(0,len(eval_dict['k_coef']))], np.array(eval_dict['k_coef'])[:,i], color='C%s'%i)
            plt.title('K Coefficient User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('K Coef')
            plt.savefig(eval_dir + 'k_coef_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(eval_dict['overfill'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(np.array(eval_dict['overfill'])[:,i], color='C%s'%i)
            plt.title('Overfilling User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('k * v^2 * d^2')
            plt.savefig(eval_dir + 'overfilling_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(eval_dict['energy'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(np.array(eval_dict['energy'])[:,i], color='C%s'%i)
            plt.title('Energy Consumption User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('Energy (Joule)')
            plt.savefig(eval_dir + 'energy_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(eval_dict['blackborder'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(np.array(eval_dict['blackborder'])[:,i], color='C%s'%i)
            plt.title('Black Border User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('Black border')
            plt.savefig(eval_dir + 'blackborder_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        fig, ax = plt.subplots()
        plt.plot(np.array(eval_dict['reward']), c='C0')
        # plt.fill_between([i for i in range(0,len(rwd_df))], rwd_df['std_high'], rwd_df['std_low'], alpha=0.2)
        plt.title('System Reward UEs=%s RBs=%s' % (nu, SUB_NUM))
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.savefig(eval_dir + 'reward_ue%s_rb%s.png' %(nu, SUB_NUM))
        # plt.show()
        plt.close(fig)


    if plot:
        train_dir = result_dir +'train/'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        for i in range(0, len(result_dict['x'][0])):
            fig, ax = plt.subplots()
            plt.scatter([j for j in range(0,len(result_dict['x']))], np.array(result_dict['x'])[:,i], color='C%s'%i)
            plt.title('Subchannel Allocation User %s' %str(i+1))
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Timestep')
            plt.ylabel('Subchannel')
            plt.savefig(train_dir + 'subchannel_allocation_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['k_coef'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.scatter([j for j in range(0,len(result_dict['k_coef']))], np.array(result_dict['k_coef'])[:,i], color='C%s'%i)
            plt.title('K Coefficient User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('K Coef')
            plt.savefig(train_dir + 'k_coef_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['overfill'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(ma(np.array(result_dict['overfill'])[:,i], n), color='C%s'%i)
            plt.title('Overfilling User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('k * v^2 * d^2')
            plt.savefig(train_dir + 'overfilling_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['energy'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(ma(np.array(result_dict['energy'])[:,i], n), color='C%s'%i)
            plt.title('Energy Consumption User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('Energy (Joule)')
            plt.savefig(train_dir + 'energy_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['blackborder'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(ma(np.array(result_dict['blackborder'])[:,i], n), color='C%s'%i)
            # plt.plot(np.array(result_dict['blackborder'])[:,i], color='C%s'%i)
            plt.title('Black Border User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('Black border')
            plt.savefig(train_dir + 'blackborder_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        # for i in range(0, len(result_dict['nooverfill_perim'][0])):
        #     fig, ax = plt.subplots()
        #     ax.grid(True)
        #     plt.plot(ma(np.array(result_dict['nooverfill_perim'])[:,i],n), color='b', label='No overfilling')
        #     plt.plot(ma(np.array(result_dict['algorithm_perim'])[:,i],n), color='C%s'%i, label='Algorithm')
        #     plt.plot(ma(np.array(result_dict['optimal_perim'])[:,i],n), color='r', label='Optimal')
        #     plt.title('Overfill Perimeter User %s' %str(i+1))
        #     plt.xlabel('Timestep')
        #     plt.ylabel('Perimeter (Pixels)')
        #     plt.legend(loc='upper right')
        #     plt.savefig(result_dir + 'perim_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
        #     # plt.show()
        #     plt.close(fig)

        fig, ax = plt.subplots()
        plt.plot(rwd_df['mean'], c='C0')
        plt.fill_between([i for i in range(0,len(rwd_df))], rwd_df['std_high'], rwd_df['std_low'], alpha=0.2)
        plt.title('System Reward UEs=%s RBs=%s' % (nu, SUB_NUM))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(train_dir + 'reward_ue%s_rb%s.png' %(nu, SUB_NUM))
        # plt.show()
        plt.close(fig)
    return avg_rwd, avg_e

print(run(300, 300, 3, 3, True, 'result/test_eval/'))

# run(EPISODES, TIMESTEP, RB, UE)