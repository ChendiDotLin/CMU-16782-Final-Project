"""
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import os, sys
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sys.path.append("../planner")
from rrt_gym import RRT

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer
from math import pi

EPSILON = 0.2
MU = 0
SIGMA = 0.05

HER = False

class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = MU
        self.sigma = SIGMA
        self.epsilon = EPSILON

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, 1)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def evaluate(env, actor, num_episodes, full_output=True):
    """Evaluate the policy. Noise is not added during evaluation.

    Args:
        num_episodes: (int) number of evaluation episodes.
    Returns:
        success_rate: (float) fraction of episodes that were successful.
        average_return: (float) Average cumulative return.
    """
    test_rewards = []
    success_vec = []
    plt.figure(figsize=(12, 12))
    for i in range(num_episodes):
        s_vec = []
        state = env.reset()
        s_t = np.array(state)
        total_reward = 0.0
        done = False
        step = 0
        success = False
        while not done:
            s_vec.append(s_t)
            a_t = actor.predict(np.reshape(s_t, (1, actor.s_dim)))[0]
            new_s, r_t, done, info = env.step(1 * (a_t > 0))
            if done and "goal" in info:
                success = True
            new_s = np.array(new_s)
            total_reward += r_t
            s_t = new_s
            step += 1
        success_vec.append(success)
        test_rewards.append(total_reward)
        if i < 9:
            plt.subplot(3, 3, i+1)
            s_vec = np.array(s_vec)
            pusher_vec = s_vec[:, :2]
            puck_vec = s_vec[:, 2:4]
            goal_vec = s_vec[:, 4:]
            plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
            plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
            plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
            plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
            plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                             color='g' if success else 'r')
            plt.xlim([-1, 6])
            plt.ylim([-1, 6])
            if i == 0:
                plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
            if i == 8:
                # Comment out the line below to disable plotting.
#                    plt.show()
                plt.close()
                pass
    if full_output:
        return success_vec, test_rewards
    return np.mean(success_vec), np.mean(test_rewards)
# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    eval_success = []
    eval_reward = []
    td_losses = []
    q_maxes = []

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        store_states = []
        store_next_states = []
        store_rewards = []
        store_actions = []
        store_done = []
        td_loss_epi = []
        q_max_epi = []

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor_noise(actor.predict(np.reshape(s, (1, actor.s_dim))))

            s2, r, terminal, info = env.step(1 * (a[0] > 0))

            store_states.append(s)
            store_actions.append(a[0])
            #store_rewards.append(reward)
            #store_next_states.append(next_state)
            #store_done.append(done)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, td_loss, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                # print(td_loss)
                ep_ave_max_q += np.amax(predicted_q_value)
                td_loss_epi.append(td_loss)
                q_max_epi.append(ep_ave_max_q)
                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            td_losses.append(np.mean(td_loss_epi))
            q_maxes.append(np.mean(q_max_epi))

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:.3f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, \
                                                                             i, (ep_ave_max_q / float(j))))
                break

        if HER:
            store_states.append(s)
            her_states, her_rewards = env.apply_hindsight(np.copy(store_states))
            total_r = 0
            n_step = 0

            for t in range(len(store_actions)):
                state = her_states[t]
                action = store_actions[t]
                reward = her_rewards[t]
                total_r += reward
                next_state = her_states[t + 1]

                box_pos = state[2:4]
                goal_pos = state[4:6]
                # check if reached
                done = np.linalg.norm(np.array(box_pos) - np.array(goal_pos)) < 0.7
                if n_step == 0 and done:
                    break
                replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                  done, np.reshape(next_state, (actor.s_dim,)))
                # self.buffer.add(state, action, reward, next_state, done)
                n_step += 1
                if done:
                    break
            print("Add her, n_step: {}, total r: {}".format(n_step, total_r))

        del store_states, store_actions
        store_states, store_actions = [], []

        if i % 30 == 0:
            success_vec, rewards_vec = evaluate(env, actor, 10)
            eval_success.append(success_vec)
            eval_reward.append(rewards_vec)
            print('\tEvaluation: success = %.2f; return = %.2f' % (np.mean(success_vec), np.mean(rewards_vec)))
            np.savez("logs/ddpg_HER={}_time=baseline.npz".format(False),
                     successes=eval_success, mean_rewards=eval_reward)
            plt.figure()
            mean = np.array([np.mean(_) for _ in eval_reward])
            std = np.array([np.std(_) for _ in eval_reward])
            suc = [np.mean(_) for _ in eval_success]
            plt.subplot(211)
            plt.plot(np.arange(1, len(eval_reward) + 1) * 100, mean, label='Mean Rewards')
            plt.title("Evaluation Rewards")
            #                plt.errorbar(np.arange(1, len(eval_reward)+1) * 50, mean, yerr=std)
            plt.fill_between(np.arange(1, len(eval_reward) + 1) * 100, mean - std, mean + std, color='g', alpha=0.5)
            plt.subplot(212)
            plt.title("Evaluation Success Rate")
            plt.plot(np.arange(1, len(eval_reward) + 1) * 100, suc)
            plt.savefig('logs/eval_Baseline_HER={}_iter={}.png'.format(HER, i))
            plt.close()
            # plot td and qmax
            plt.figure()
            plt.subplot(211)
            plt.plot(td_losses)
            plt.title('TD Loss')
            plt.subplot(212)
            plt.plot(q_maxes)
            plt.title('Q Max')
            plt.savefig('logs/loss_Baseline_HER={}_iter={}.png'.format(HER, i))
            plt.close()

def main(args):
    with tf.Session() as sess:
        start = [pi / 2, pi / 4, pi / 2, pi / 4, pi / 2]
        goal = [pi / 8, 3 * pi / 4, pi, 0.9 * pi, 1.5 * pi]

        env = RRT("../planner/map1.txt", start, goal)

        # np.random.seed(int(args['random_seed']))
        # tf.set_random_seed(int(args['random_seed']))
        # env.seed(int(args['random_seed']))

        state_dim = len(env.state)
        action_dim = 1
        # Ensure action bound is symmetric
        #assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = EpsilonNormalActionNoise()#OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        train(sess, env, args, actor, critic, actor_noise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.98)
    parser.add_argument('--tau', help='soft target update parameter', default=0.05)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=1024)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
