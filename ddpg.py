# -*- coding: utf-8 -*-


import os
import random
import gym
import time
from collections import deque

import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Lambda, concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import pandas as pd
from environment import TrafficPrediction

class DDPG():
    """Deep Deterministic Policy Gradient Algorithms.
    """
    def __init__(self):
        super(DDPG, self).__init__()

        self.sess = K.get_session()
        self.env = TrafficPrediction()
        self.bound = 1.0

        # update rate for target model.
        self.TAU = 0.01
        # experience replay.
        self.memory_buffer = deque(maxlen=3000)
        # discount rate for q value.
        self.gamma = 0.95
        # epsilon of action selection
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01

        # actor learning rate
        self.a_lr = 0.00001
        # critic learining rate
        self.c_lr = 0.0001

        # ddpg model
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # target model
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        # gradient function

        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

        if os.path.exists('model_/ddpg_actor.h5') and os.path.exists('model_/ddpg_critic.h5'):
            self.actor.load_weights('model_/ddpg_actor.h5')
            self.critic.load_weights('model_/ddpg_critic.h5')

    def _build_actor(self):
        """Actor model.
        """
        traffic_input = Input(shape=(self.env.inputstep, 257), name='traffic_input')
        x = LSTM(128, activation='tanh')(traffic_input)
        error_input = Input(shape=(257,), name='error_input')
        x = concatenate([x, error_input])
        # x = Dense(40, activation='relu')(inputs)
        x = Dense(200, activation='relu')(x)
        output = Dense(2, activation='sigmoid')(x)
        # output = Lambda(lambda x: x * self.bound)(x)

        model = Model(inputs=[traffic_input, error_input], outputs=output)
        # print(model.summary())
        model.compile(loss='mse', optimizer=Adam(lr=self.a_lr))

        return model

    def _build_critic(self):
        """Critic model.
        """
        traffic_input = Input(shape=(self.env.inputstep, 257), name='state_input')
        error_input = Input(shape=(257,), name='traffic_input')
        action_input = Input(shape=(2,), name='action_input')
        # s = Dense(40, activation='relu')(sinput)
        # a = Dense(40, activation='relu')(ainput)
        x = LSTM(128, activation='tanh')(traffic_input)
        x = concatenate([x, error_input])
        x = Dense(200, activation='relu')(x)
        x = concatenate([x, action_input])
        x = Dense(200, activation='relu')(x)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=[traffic_input, error_input, action_input], outputs=output)
        # print(model.summary())
        model.compile(loss='mse', optimizer=Adam(lr=self.c_lr))

        return model

    def actor_optimizer(self):
        """actor_optimizer.

        Returns:
            function, opt function for actor.
        """
        self.ainput = self.actor.input # Return: input tensor or list of input tensors.
        # print('ainput:',self.ainput)
        # print('ainput type:',type(self.ainput))
        # print('ainput shape:', np.asarray(self.ainput).shape)
        self.ainput_traffic = self.ainput[0]
        self.ainput_error = self.ainput[1]
        
        aoutput = self.actor.output
        
        trainable_weights = self.actor.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, 2))

        # tf.gradients will calculate dy/dx with a initial gradients for y
        # action_gradient is dq / da, so this is dq/da * da/dparams
        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        self.opt = tf.train.AdamOptimizer(self.a_lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def critic_gradient(self):
        """get critic gradient function.

        Returns:
            function, gradient function for critic.
        """
        cinput = self.critic.input # [traffic_input, error_input, action_input]
        coutput = self.critic.output

        # compute the gradient of the action with q value, dq/da.
        action_grads = K.gradients(coutput, cinput[2])

        return K.function([cinput[0], cinput[1], cinput[2]], action_grads)

    def OU(self, x, mu=0, theta=0.15, sigma=0.2):
        """Ornstein-Uhlenbeck process.
        formula：ou = θ * (μ - x) + σ * w

        Arguments:
            x: action value.
            mu: μ, mean fo values.
            theta: θ, rate the variable reverts towards to the mean. 
            sigma：σ, degree of volatility of the process.

        Returns:
            OU value
        """
        return np.asarray(list(map(lambda x_: theta * (mu - x_) + sigma * np.random.randn(1), x)))

    def egreedy_action(self, X):
        """ε-greedy选择action

        Arguments:
            state: 状态

        Returns:
            action: 动作
        """
        if np.random.rand() <= self.epsilon:
            #print('random action:_')
            return random.randint(0, 1)
        else:
            return self.get_action(X)
   
    def get_action(self, X):
        """get actor action with ou noise.
        At same time, ε-greedy is used to selected actions.

        Arguments:
            X: state value (traffic_state, error_state).
        """
        action = self.actor.predict([np.asarray(X[0]).reshape(-1, self.env.inputstep, 257), np.asarray(X[1]).reshape(-1, 257)])
        action = action.reshape(2,)

        # add randomness to action selection for exploration
        noise = max(self.epsilon, 0) * self.OU(action).reshape(2,)
        action = np.clip(action + noise, 0, self.bound)

        return action

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.

        Arguments:
            state: observation.
            action: action.
            reward: reward.
            next_state: next_observation.
            done: if game done.
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """update epsilon.
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """process batch data.

        Arguments:
            batch: batch size.

        Returns:
            states: states.
            actions: actions.
            y: Q_value.
        """
        y = []
         # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        states = np.array([d[0] for d in data])
        # print('states shape:',states.shape)
        traffic_state = np.array([s[0] for s in states])
        # print('traffic_state shape:',traffic_state.shape)
        error_state = np.array([s[1] for s in states])
        # print('error_state shape:', error_state.shape)
        actions = np.array([d[1] for d in data])
        # print('actions shape in process_batch:', actions.shape)
        next_states = np.array([d[3] for d in data])
        
        next_traffic = np.array([n[0] for n in next_states]).reshape(-1, self.env.inputstep, 257)
        next_error = np.array([n[1] for n in next_states]).reshape(-1, 257)
        # print('next_error shape:', next_error.shape)

        # Q_target。
        next_actions = self.target_actor.predict([next_traffic, next_error])
        q = self.target_critic.predict([next_traffic, next_error, np.asarray(next_actions).reshape(-1, 2)])

        # update Q value
        for i, (_, _, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * q[i][0]
            y.append(target)

        return traffic_state, error_state, actions, y

    def update_model(self, X1_traffic, X1_error, X2, y):
        """update ddpg model.

        Arguments:
            states: states.
            actions: actions.
            y: Q_value.

        Returns:
            loss: critic loss.
        """
#        loss = self.critic.train_on_batch([X1, X2], y)
        loss = self.critic.fit([X1_traffic, X1_error, X2], y, verbose=0)
        loss = np.mean(loss.history['loss'])

        X3 = self.actor.predict([X1_traffic, X1_error])
        a_grads = np.array(self.get_critic_grad([X1_traffic, X1_error, X3]))[0]
        # print('a_grads:',type(a_grads))
        # print('a_grads shape:',a_grads.shape)
        # print('X1_traffic shape', X1_traffic.shape)
        # print('X1_error shape', X1_error.shape)        
        self.sess.run(self.opt, feed_dict={
            self.ainput_traffic: X1_traffic,
            self.ainput_error: X1_error,
            self.action_gradient: a_grads
        })

        return loss

    def update_target_model(self):
        """soft update target model.
        formula：θ​​t ← τ * θ + (1−τ) * θt, τ << 1. 
        """
        critic_weights = self.critic.get_weights()
        actor_weights = self.actor.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]

        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]

        self.target_critic.set_weights(critic_target_weights)
        self.target_actor.set_weights(actor_target_weights)

    def train(self, episode, batch):
        """training model.
        Arguments:
            episode: ganme episode.
            batch： batch size of episode.

        Returns:
            history: training history.
        """
        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        for i in range(episode):
            observation = self.env.reset(data_type='vali')
            self.env.pointer += 1
            reward_sum = 0
            losses = []

            for j in range(100):
                # chocie action from ε-greedy.
                x = observation

                # actor action
                action = self.get_action(x)
                # print('action:', action.shape) (257,)
                observation, reward, done, _ = self.env.step(action, self.env.pointer, data_type='vali')
                # add data to experience replay.
                reward_sum += reward
                self.remember(x, action, reward, observation, done)

                if len(self.memory_buffer) > batch:
                    X1_traffic, X1_error, X2, y = self.process_batch(batch) #return: states, actions, y
                    # print('X1_traffic:', X1_traffic.shape) ()
                    # print('X1_error:', X1_error.shape)
                    # print('X2:', X2.shape)
                    # print('y:', np.array(y).shape)
                    
                    # update DDPG model
                    loss = self.update_model(X1_traffic, X1_error, X2, np.asarray(y))
                    # update target model
                    self.update_target_model()
                    # reduce epsilon pure batch.
                    self.update_epsilon()

                    losses.append(loss)
                
                self.env.pointer += 1

            loss = np.mean(losses)
            history['episode'].append(i)
            history['Episode_reward'].append(reward_sum)
            history['Loss'].append(loss)

            print('Episode: {}/{} | reward: {:.4f} | loss: {:.3f}'.format(i, episode, reward_sum, loss))

        self.actor.save_weights('model_/ddpg_actor.h5')
        self.critic.save_weights('model_/ddpg_critic.h5')

        return history

    def play(self, episode):
        """play game with model.
        """
        print('play...')
        observation = self.env.reset(data_type='test')
        self.env.pointer += 1

        reward_sum = 0
        # random_episodes = 0

        # while random_episodes < 100:
        while self.env.pointer < (len(self.env.testY_nofilt)-1):
            # self.env.render()
            
            # print('pointer at: ', self.env.pointer)
            x = observation
            x_traffic = np.array(x[0]).reshape(-1, self.env.inputstep, 257)
            x_error = np.array(x[1]).reshape(-1, 257)
            action = self.actor.predict([x_traffic, x_error]).reshape(2,)
            observation, reward, done, _ = self.env.step(action, self.env.pointer, data_type='test')

            reward_sum += reward
            self.env.pointer += 1
        
        print('\n##########')
        RMSE, MAE, MAPE = self.env.get_error(np.array(self.env.set_predY).reshape(-1, 257),np.array(self.env.set_realY).reshape(-1, 257))
        print('Final TBSM: RMSE=%.4f,MAE=%.4f,MAPE=%.4f'%(RMSE,MAE,MAPE))
        RMSE_, MAE_, MAPE_ = self.env.get_error(np.array(self.env.set_predY_).reshape(-1, 257),np.array(self.env.set_realY).reshape(-1, 257))
        print('Final ITBSM: RMSE=%.4f,MAE=%.4f,MAPE=%.4f'%(RMSE_,MAE_,MAPE_))
        with open('model_/log.txt','a') as f:
            f.write('\n*** Episode=%d ***'%episode)
            f.write('\n TBSM: RMSE=%.4f,MAE =%.4f,MAPE =%.4f'%(RMSE,MAE,MAPE))
            f.write('\nITBSM: RMSE=%.4f,MAE_=%.4f,MAPE_=%.4f'%(RMSE_,MAE_,MAPE_))
        
        df = pd.DataFrame(np.array([i * self.env.maxv for i in self.env.testY_nofilt]).reshape(-1, 257))
        df.to_csv('model_/%d-testY%dp%d.csv'%(episode, self.env.inputstep, self.env.predstep), header=False, index=False)
        
        df = pd.DataFrame(np.array(self.env.set_realY).reshape(-1, 257))
        df.to_csv('model_/%d-realY%dp%d.csv'%(episode, self.env.inputstep, self.env.predstep), header=False, index=False)
        
        df = pd.DataFrame(np.array(self.env.set_predY).reshape(-1, 257))
        df.to_csv('model_/%d-tbsm-predY%dp%d.csv'%(episode, self.env.inputstep, self.env.predstep), header=False, index=False)
        
        df = pd.DataFrame(np.array(self.env.set_predY_).reshape(-1, 257))
        df.to_csv('model_/%d-reinforced-predY%dp%d.csv'%(episode, self.env.inputstep, self.env.predstep), header=False, index=False)
        
        self.env.close()

    def save_history(self, history, name):
        name = os.path.join('history', name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
        
    def save_params(self, params, name):
        name = os.path.join('params', name)
        df = pd.DataFrame.from_dict(params)
        df.to_csv(name, index=False, encoding='utf-8')

if __name__ == '__main__':
    model = DDPG()
    
    training_time = []
    for i in range(0,31):
        episode = 200*i
        start_time = time.time()
        history = model.train(200, 256)
        training_time.append([i,time.time()-start_time])
        model.save_history(history, 'H:/AutoKNN/AutoKNN/adaptive-model/model_/%d-ddpg.csv'%episode)
        if i%2==0:
            model.play(episode)
            model.save_params(model.env.predictor.params_record, 'H:/AutoKNN/AutoKNN/adaptive-model/model_/%d-params.csv'%episode)
        print('training time for this 200*episode is',training_time[-1])
#    model.play(9999)