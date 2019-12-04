import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import backend as K
import gym

from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n

        model_adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        critic_adam = keras.optimizers.Adam(lr=critic_lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model.compile(loss='categorical_crossentropy',optimizer=model_adam,metrics=['accuracy'])
        self.critic_model.compile(loss='mean_squared_error',optimizer=critic_adam,metrics=['accuracy'])
        
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def discounted_reward(self,rewards,Vs,gamma):
        T = len(rewards)
        Rt = [0]*T
        for t in reversed(range(T)):
            Vend = 0 if t+self.n >= T else Vs[t+self.n]
            Rt[t] = (gamma**self.n)*Vend
            for k in range(self.n):
                if t+k < T:
                    Rt[t] += (gamma**k)*rewards[t+k]
        return Rt

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states,actions,rewards = self.generate_episode(env)
        Vs = self.critic_model.predict(np.array(states),batch_size=1)
        Rt = self.discounted_reward(rewards,Vs,gamma)
        Y_actor = np.zeros((len(states),env.action_space.n))
        for i in range(len(states)):
            Y_actor[i][actions[i]] = (Rt[i] - Vs[i])
        actor_histyory = self.model.fit(np.array(states),Y_actor,epochs=1,batch_size=len(rewards),verbose=0)
        critic_history = self.critic_model.fit(np.array(states),np.array(Rt),epochs=1,batch_size=len(rewards),verbose=0)
        return

    def generate_episode(self, env, render=False):
         # Generates an episode by executing the current policy in the given env.
         # Returns:
         # - a list of states, indexed by time step
         # - a list of actions, indexed by time step
         # - a list of rewards, indexed by time step
         # TODO: Implement this method.
         states = []
         actions = []
         rewards = []
         done = False
         state = env.reset()

         while not done:
             states.append(state)
             prediction = self.model.predict(state.reshape((1,-1)),batch_size=1)[0]
             action = np.random.choice(len(prediction),p=prediction)
             actions.append(action)

             state,reward,done,info = env.step(action)
             rewards.append(reward/100) # down scale the reward here

         return states, actions, rewards

    def test(self, env, model_file=None):

        rewards = []

        for epi in range(100):
            done = False
            state = env.reset()
            total_r = 0
            while not done:
                prediction = self.model.predict(np.expand_dims(state, axis=0))[0]
                action = np.argmax(prediction)
                state, reward, done, _ = env.step(action)
                total_r += reward
            rewards.append(total_r)

        mean = np.mean(rewards)
        std = np.std(rewards)
        return mean,std

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def net(env):
    model = Sequential()
    model.add(Dense(units=32,
                   kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',distribution='uniform',
                                                                   seed=None),
                   bias_initializer='zeros',
                   activation='relu'))

#    model.add(BatchNormalization())

    model.add(Dense(units=16,
                   kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform',
                                                                   seed=None),
                   bias_initializer='zeros',
                   activation='relu'))

#    model.add(BatchNormalization())

    model.add(Dense(units=16,
                   kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform',
                                                                   seed=None),
                   bias_initializer='zeros',
                   activation='relu'))

#    model.add(BatchNormalization())

    model.add(Dense(units=env.action_space.n,
                   kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform',
                                                                   seed=None),
                   bias_initializer='zeros',
                   activation='softmax'))
    return model
    
def critic_net(env):
    model = Sequential()
    model.add(Dense(units=16,
                   kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',distribution='uniform',
                                                                   seed=None),
                   bias_initializer='zeros',
                   activation='relu'))
    model.add(Dense(units=16,
                   kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform',
                                                                   seed=None),
                   bias_initializer='zeros',
                   activation='relu'))
    model.add(Dense(units=1))
    return model

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
#    env = gym.make('CartPole-v0')

    # TODO: Create the model.
    model = net(env)
    critic_model = critic_net(env)
    # TODO: Train the model using A2C and plot the learning curves.
    gamma = 0.99
    a2c = A2C(model,lr,critic_model,critic_lr)
    
    k = 500

    mean_vec = []
    std_vec = []
    epi_vec = []

    start_time = time()

    for epi in range(num_episodes):
        a2c.train(env,gamma)
        if epi%k==0:
            reward_vec = []
#            for _ in range(100):
#                _,_,rewards = a2c.generate_episode(env)
#                reward_vec.append(np.sum(rewards)*100) # upscale the reward back
            mean, std = a2c.test(env)
            
            mean_vec.append(mean)
            std_vec.append(std)
            epi_vec.append(epi)
            print('{}/{} Finished. Mean reward: {}, std: {}, remaining time: {}min'.format(epi, num_episodes, mean, std, (num_episodes-epi-1)*(time()-start_time)/(epi+1)/60))

            fig = plt.figure(figsize=(8,4))
            plt.title('Cumulative Reward')
            plt.errorbar(epi_vec,mean_vec,std_vec)
            plt.xlabel('episodes')
            plt.ylabel('rewards')

            plt.savefig('plots/reward_for_A2C_N={}_at_{}.png'.format(n, epi), dpi=300)

    fig = plt.figure(figsize=(8,4))
    plt.title('Cumulative Reward')
    plt.errorbar(epi_vec,mean_vec,std_vec)
    plt.xlabel('episodes')
    plt.ylabel('rewards')

    plt.savefig('reward_for_A2C_N=%i.png' %n, dpi=300)
    plt.show()
    

if __name__ == '__main__':
    main(sys.argv)
