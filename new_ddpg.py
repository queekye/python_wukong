"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import time
from env_zeromap_wukong import Env
from ou_noise import OUNoise


#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS = 200
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 1000000
REPLAY_START = 1000
BATCH_SIZE = 32


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim.sum() + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim.sum()]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            layer_1 = tf.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
            layer_2 = tf.layers.dense(layer_1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            a1 = tf.layers.dense(layer_2, self.a_dim[0], activation=tf.nn.tanh, name='a1', trainable=trainable)
            a1_norm = tf.nn.l2_normalize(a1, dim=-1)
            a2 = tf.layers.dense(layer_2, self.a_dim[1], activation=tf.nn.tanh, name='a2', trainable=trainable)
            a = tf.concat([a1_norm, a2], axis=-1)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim.sum(), n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            layer_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            layer_2 = tf.layers.dense(layer_1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            return tf.layers.dense(layer_2, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################


env = Env()

s_dim = env.s_dim
a_dim = env.a_dim
a_bound = env.a_bound

ddpg = DDPG(a_dim, s_dim, a_bound)
exploration_noise = OUNoise(a_dim.sum())  # control exploration
t1 = time.time()
replay_num = 0
for i in range(MAX_EPISODES):
    t_start = time.time()
    sd = i * 3 + 100
    s = env.set_state_seed(sd)
    exploration_noise.reset()
    ep_reward = 0
    ave_w = 0
    j = 0
    r = 0
    for j in range(MAX_EP_STEPS):
        # Add exploration noise
        a = ddpg.choose_action(s)
        ave_w += np.linalg.norm(a[-a_dim[1]:])
        a += exploration_noise.noise()  # add randomness to action selection for exploration
        a[:a_dim[0]] /= max(np.linalg.norm(a[:a_dim[0]]), 1e-8)
        a = np.minimum(a_bound, np.maximum(-a_bound, a))

        s_, r, done, controlled, real_a = env.step(a)
        if not controlled:
            # agent.perceive(state_input, action, -1, next_state, True)
            ddpg.store_transition(s, real_a, r, s_)
        else:
            ddpg.store_transition(s, a, r, s_)
        replay_num += 1
        if ddpg.pointer > REPLAY_START:
            ddpg.learn()

        s = s_
        ep_reward += r

        if done:
            break
    ave_w /= j+1
    print("episode: %10d   ep_reward:%10.5f   last_reward:%10.5f   replay_num:%10d   "
          "cost_time:%10.2f    ave_w:" % (i, ep_reward, r, replay_num, time.time() - t_start), ave_w)

print('Running time: ', time.time() - t1)
