import scipy.io as sio
import numpy as np
import random as rn
import time as tm
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

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

def run(t, s, ue, plot, result_dir):

    # This algorithm generates K modes from DNN, and chooses with largest
    # reward. The mode with largest reward is stored in the memory, which is
    # further used to train the DNN.
    print("DQN Computing Phase")
    nu = ue  # number of users
    n = t  # number of time frames
    b = 1 * 10 ** 6
    SUB_NUM = s

    NUM_EPISODES = 100
    INITIAL_EPSILON = 0.4
    FINAL_EPSILON = 0.01
    EPSILON_DECAY = 3*n*NUM_EPISODES/4
    TRAINING_CYCLE = 10
    TARGET_UPDATE_CYCLE = 1
    epsilon = INITIAL_EPSILON

    RESULT_DIR = datetime.now().strftime("%y%m%d_%H%M")

    playout_d = [0.01, 0.05, 0.1, 0.2, 0.3] 

    # Load data
    channel = sio.loadmat('data/ch_gain/data_%d' % 10)['input_h']
    channel = channel * 1000000

    v_df = pd.read_csv('data/ang_vec/ang_vec_3users.csv')
    v_df = v_df.clip(-4, 4)

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

    all_subs = [x / 1000000 for x in all_sub]    # all_subs used for computation rate calculation
    split_idx = int(len(adj_channel))//SUB_NUM  # training data indices

    discrete_action = True

    if discrete_action:
        out_size = pow(SUB_NUM+1, nu) * pow(len(playout_d), nu)

    network = DeepQNetwork(net=[(nu,2), 120, 80, out_size], # Input layer (nu), 120 and 80 neurons of hidden layers and output layer (nu)
                    learning_rate=0.01,
                    batch_size=64,
                    training_cycle = TRAINING_CYCLE,
                    target_update_cycle = TARGET_UPDATE_CYCLE,
                    sub_num = SUB_NUM,
                    enable_DDQN = False
                    )

    action_space = [SUB_NUM if i%2 else len(playout_d)-1 for i in range(1,2*nu+1)]
    # print(action_space)
    possible_actions = [list(range(0, (k + 1))) for k in action_space]

    # print(possible_actions)
    action_map = generate_actions(possible_actions)[0]

    # print(len(action_map))

    total_rwd = []

    result_dict = defaultdict(list)

    time_duration = []

    # v_array = [rn.uniform(10**-4, 1) for i in range(0,nu)]   # randomize head velocity for initial testing

    for episode in range(NUM_EPISODES):
        episode_reward = 0
        episode_q = 0
        # state = env.reset()

        for i in range(n-2):  # Time frames looping
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            start_time = tm.time()
            
            i_idx = (i + (episode * n)) % split_idx

            v_array = list(v_df.iloc[i])
            # print(v_array)

            # v_array = [rn.uniform(10**-4, 1) for i in range(0,nu)]
            
            state = (all_sub[0][i_idx,:], v_array)                             # state = channel gain + head velocity

            state = tf.expand_dims(state,0)
            # print(state)

            if np.random.uniform() < epsilon:
                # print('EPSILON')
                action = rn.randint(0,out_size-1)
            else:
                # print('NETWORK')
                tmp = network.evaluation_network(state)
                # print(tmp)
                action = tf.argmax(tmp[0])
            # print(action)
            action_q = action_map[action]
            # print(action_q)
            env_dict = step(i_idx, action_q, SUB_NUM, ue, b, all_subs, v_array)
            # print(env_dict['overfill'])
            reward = -sum(env_dict['energy'])
            # print(reward)
            # print(i)
            v_array = list(v_df.iloc[i+1])
            next_state = (all_sub[0][i_idx+1,:], v_array)
            next_state = tf.expand_dims(next_state,0)

            network.append_experience({'state':state,
            'action':[action],'reward':[reward],'next_state': next_state})
            episode_reward += reward
            episode_q += sum(env_dict['rate'])

            if network.training_counter >= network.training_cycle:
                network.train()
                network.delete_experience()

            duration = tm.time()-start_time
            time_duration.append(duration)

            for k, v in env_dict.items():
                result_dict[k].append(v)
                result_dict['sum_' + k].append(sum(v))

        print('Episode {} finished after {} timesteps, total rewards {}, epsilon {}'.format(episode, t+1, episode_reward, epsilon))
        total_rwd.append(episode_reward/n)

    avg_ue_r = np.mean(result_dict['rate'], axis=0)
    avg_ue_e = np.mean(result_dict['energy'], axis=0)

    avg_r = sum(result_dict['sum_rate'])/(NUM_EPISODES * n)
    avg_e = sum(result_dict['sum_energy'])/(NUM_EPISODES * n)

    rwd_df = pd.DataFrame(total_rwd)
    rolling_mean = rwd_df.rolling(5).mean()
    rolling_std = rwd_df.rolling(5).std()
    rwd_df['mean'] = rolling_mean
    rwd_df['std_high'] = rolling_mean + rolling_std
    rwd_df['std_low'] = rolling_mean - rolling_std

    if plot:
        for i in range(0, len(result_dict['x'][0])):
            fig, ax = plt.subplots()
            plt.scatter([j for j in range(0,len(result_dict['x']))], np.array(result_dict['x'])[:,i], color='C%s'%i)
            plt.title('Subchannel Allocation User %s' %str(i+1))
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Timestep')
            plt.ylabel('Subchannel')
            plt.savefig(result_dir + 'subchannel_allocation_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['delay'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.scatter([j for j in range(0,len(result_dict['delay']))], np.array(result_dict['delay'])[:,i], color='C%s'%i)
            plt.title('Playout Delay User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('Delay (second)')
            plt.savefig(result_dir + 'playout_delay_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['overfill'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(ma(np.array(result_dict['overfill'])[:,i], n), color='C%s'%i)
            plt.title('Overfilling User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('k * v^2 * d^2')
            plt.savefig(result_dir + 'overfilling_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        for i in range(0, len(result_dict['delay_bound'][0])):
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.plot(ma(np.array(result_dict['delay_bound'])[:,i], n), color='C%s'%i)
            plt.title('Delay Bound User %s' %str(i+1))
            plt.xlabel('Timestep')
            plt.ylabel('Delay Bound')
            plt.savefig(result_dir + 'delay_bound_rb%s_user%s.png' %(SUB_NUM, str(i+1)))
            # plt.show()
            plt.close(fig)

        fig, ax = plt.subplots()
        plt.plot(rwd_df['mean'], c='C0')
        plt.fill_between([i for i in range(0,len(rwd_df))], rwd_df['std_high'], rwd_df['std_low'], alpha=0.2)
        plt.title('VR Energy Consumption UEs=%s RBs=%s' % (nu, SUB_NUM))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(result_dir + 'energy_consumption_ue%s_rb%s.png' %(nu, SUB_NUM))
        # plt.show()
        plt.close(fig)
    return avg_e, avg_ue_e

print(run(300, 1, 3, True, 'result/test/'))