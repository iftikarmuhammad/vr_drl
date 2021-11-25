import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model


class DeepQNetwork(Model):
    def __init__(
        self,
        net,
        learning_rate,
        batch_size,
        training_cycle, 
        target_update_cycle, 
        sub_num,
        enable_DDQN): 
        super(DeepQNetwork, self).__init__()
        self.net = net  # the size of the DNN
        self.lr = learning_rate
        self.batch_size = batch_size
        self.opt = keras.optimizers.Adam(learning_rate)
        self.sub_num = sub_num
        
        # Q-learning
        self.gamma = 0.9
        self.tau = 0.001
        self.loss_function = keras.losses.MeanSquaredError()

        self.training_counter = 0
        self.training_cycle = training_cycle
        self.target_update_counter = 0
        self.target_update_cycle = target_update_cycle
        self.memories_nameList = ['state','action','reward','next_state']
        self.memories_dict = {}
        for itemname in self.memories_nameList:
            self.memories_dict[itemname] = None

        self.enable_DDQN = enable_DDQN

        self.evaluation_network = self.build_model()
        self.target_network = self.build_model()
        self.evaluation_network

        self.training_loss = []


    def build_model(self):
        model = keras.Sequential([
                    layers.Flatten(),
                    layers.Dense(self.net[1], activation='relu'),  # the first hidden layer
                    layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
                    layers.Dense(self.net[-1], activation='linear'),  # the output layer
                ])
        return model


    def train(self):
        random_select = np.random.choice(self.training_cycle,self.batch_size)

        states = self.memories_dict["state"][random_select]
        actions = self.memories_dict["action"][random_select]
        rewards = self.memories_dict["reward"][random_select]
        nextStates = self.memories_dict["next_state"][random_select]

        with tf.GradientTape() as tape:
            q_eval_arr = self.evaluation_network(states)
            q_eval = tf.reduce_max(q_eval_arr,axis=1)
            if self.enable_DDQN == True:
                # Double Deep Q-Network
                q_values = self.evaluation_network(nextStates)
                q_values_actions = tf.argmax(q_values,axis=1)
                target_q_values = self.target_network(nextStates)
                # discount_factor = target_q_values[range(self.batch_size),q_values_actions]
                indice = tf.stack([range(self.batch_size),q_values_actions],axis=1)
                discount_factor = tf.gather_nd(target_q_values,indice)
            else:
                # Deep Q-Network
                target_q_values = self.target_network(nextStates)
                discount_factor = tf.reduce_max(target_q_values,axis=1)
            
            # Q function
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.training_cycle)
            q_target = rewards + self.gamma * discount_factor
            loss = self.loss_function(q_eval,q_target)
            self.training_loss.append(loss)

        gradients_of_network = tape.gradient(loss,self.evaluation_network.trainable_variables)
        self.opt.apply_gradients(zip(gradients_of_network, self.evaluation_network.trainable_variables))
        self.target_update_counter += 1
        # DQN - Frozen update
        if self.target_update_counter % self.target_update_cycle == 0:
            self.target_network.set_weights(self.tau * np.array(self.evaluation_network.get_weights()) + (1 - self.tau) * np.array(self.target_network.get_weights()))

        
    def append_experience(self, dict):
        # replace the old memory with new memory
        for itemname in self.memories_nameList:
            if self.memories_dict[itemname] is None:
                self.memories_dict[itemname] = dict[itemname]
            else:
                self.memories_dict[itemname] = np.append(self.memories_dict[itemname],dict[itemname],axis=0)
        self.training_counter += 1


    def call(self):
        return


    def delete_experience(self):
        for itemname in self.memories_nameList:
            self.memories_dict[itemname] = None
        self.training_counter = 0


    def plot_cost(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.style.use('fivethirtyeight')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('w')
        fig.set_facecolor('w')
        ax.grid(True)
        plt.plot(np.arange(len(self.training_loss)), self.training_loss)
        plt.ylabel('Training loss of DNN', fontsize=20)
        plt.xlabel('Learning steps', fontsize=20)
        plt.savefig('TrainingLossFn_5.png', facecolor=fig.get_facecolor(), alpha=0.01)
        plt.show()