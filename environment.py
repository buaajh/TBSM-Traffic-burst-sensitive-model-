# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import gym
from gym import spaces
from collections import deque
from preprocess import DataPreprocess as dp
from tbsm import TBSM
import h5py
import warnings

warnings.filterwarnings('ignore')

class TrafficPrediction(gym.Env):
    def __init__(self):
        self.delta = 1.
        self.state = None
        self.predictor = TBSM()
        self.pointer = 0
        
        self.inputstep = self.predictor.dp.inputstep
        self.predstep = self.predictor.dp.predstep
        self.maxv = np.asarray(self.predictor.dp.maxv)
        
        self.action_space = spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32),#alpha & k(mapping to (0,200])
        self.observation_space = spaces.Tuple([
                spaces.Box(low=0., high=1., shape=(self.inputstep, 257), dtype=np.float32),
                spaces.Box(low=0., high=1., shape=(257,), dtype=np.float32)])
                #(last_state_point,)

        if os.path.exists('data%dp%d/valiX.h5'%(self.inputstep,self.predstep)) and os.path.exists('data%dp%d/valiY.h5'%(self.inputstep,self.predstep)) and os.path.exists('data%dp%d/valiY_nofilt.h5'%(self.inputstep,self.predstep)):
            print('loading vali data from the local...')
            with h5py.File('data%dp%d/valiX.h5'%(self.inputstep,self.predstep)) as f:
                self.valiX = f['data'][:]
            with h5py.File('data%dp%d/valiY.h5'%(self.inputstep,self.predstep)) as f:
                self.valiY = f['data'][:]
            with h5py.File('data%dp%d/valiY_nofilt.h5'%(self.inputstep,self.predstep)) as f:
                self.valiY_nofilt = f['data'][:]
        else:
            self.valiX, self.valiY, self.valiY_nofilt = self.predictor.dp.get_data(data_type='vali')
            with h5py.File('data%dp%d/valiX.h5'%(self.inputstep,self.predstep),'w') as f:
                f['data'] = self.valiX
            with h5py.File('data%dp%d/valiY.h5'%(self.inputstep,self.predstep),'w') as f:
                f['data'] = self.valiY
            with h5py.File('data%dp%d/valiY_nofilt.h5'%(self.inputstep,self.predstep),'w') as f:
                f['data'] = self.valiY_nofilt
        
        if os.path.exists('data%dp%d/testX.h5'%(self.inputstep,self.predstep)) and os.path.exists('data%dp%d/testY.h5'%(self.inputstep,self.predstep)) and os.path.exists('data%dp%d/testY_nofilt.h5'%(self.inputstep,self.predstep)):
            print('loading test data from the local...')
            with h5py.File('data%dp%d/testX.h5'%(self.inputstep,self.predstep)) as f:
                self.testX = f['data'][:]
            with h5py.File('data%dp%d/testY.h5'%(self.inputstep,self.predstep)) as f:
                self.testY = f['data'][:]
            with h5py.File('data%dp%d/testY_nofilt.h5'%(self.inputstep,self.predstep)) as f:
                self.testY_nofilt = f['data'][:]
        else:
            self.testX, self.testY, self.testY_nofilt = self.predictor.dp.get_data(data_type='test')
            with h5py.File('data%dp%d/testX.h5'%(self.inputstep,self.predstep),'w') as f:
                f['data'] = self.testX
            with h5py.File('data%dp%d/testY.h5'%(self.inputstep,self.predstep),'w') as f:
                f['data'] = self.testY
            with h5py.File('data%dp%d/testY_nofilt.h5'%(self.inputstep,self.predstep),'w') as f:
                f['data'] = self.testY_nofilt
        
        self.set_predY = []
        self.set_predY_ = []
        self.set_realY = []
        self.diff_deque = deque(maxlen=self.predstep+1)

    def get_error(self, predY, realY):
        """
        Input: predY and realY are arrays.
        """
        rmse = np.sqrt(np.mean(np.square(predY - realY)))
        mae = np.mean(np.abs(predY - realY))
        mape = np.mean(np.abs(predY - realY) / (realY)) * 100
        return rmse, mae, mape
    
    def step(self, action, pointer, data_type='vali'):
        
        if data_type == 'vali':
            self.X = self.valiX[pointer]
            self.X_next = self.valiX[pointer+1]
            self.realY = self.valiY_nofilt[pointer]
        elif data_type == 'test':
            self.X = self.testX[pointer]
            self.X_next = self.testX[pointer+1]
            self.realY = self.testY_nofilt[pointer]
        
        predY, predY_ = self.predictor.predict(self.X, action)
        self.predictor.params_record['time'].append(pointer)
        
        self.set_predY.append(predY * self.maxv)
        self.set_predY_.append(predY_ * self.maxv)
        self.set_realY.append(self.realY * self.maxv)
        
        rmse, mae, mape = self.get_error(predY * self.maxv, self.realY * self.maxv)
        # print('MAE =%.3f, MAPE =%.3f'%(mae, mape))
        rmse, mae_, mape_ = self.get_error(predY_ * self.maxv, self.realY * self.maxv)
        # print('MAE_=%.3f, MAPE_=%.3f'%(mae_, mape_))
        
        diff = (np.asarray(predY_) - np.asarray(self.realY)).reshape(self.predictor.link,)
        self.diff_deque.append(diff)
        self.diff = self.diff_deque[0]
        # print('self.diff shape (in StepFunc)',self.diff.shape)
        self.state = (self.X_next, self.diff)
        
        # done = (mae_ < (self.error[0] + 0.5 * self.delta) and mape_ < (self.error[1] + self.delta))
        # done = (mae_ <= (mae + 0.2 * self.delta)) and (mape_ <= (mape + self.delta))
        # done = (mae_ <= mae) and (mape_ <= mape)
        done  = (mae_ <= 1.01*mae) and (mape_ <= 1.01*mape)
        done = bool(done)
        
        # update mae& mape
        self.error = (mae_, mape_)
        
        if done:
            if ((mae_-mae) >= 0) or ((mape_-mape) >= 0):
                reward = 0
            else:
                reward = (max(0,100*(mae-mae_)/mae) + max(0,100*(mape-mape_)/mape)) * 0.5
        else:
            reward = ((mae-mae_)/mae + (mape-mape_)/mape)*100*0.5
        
        return self.state, reward, done, {}
    
    def reset(self, data_type='vali'):
        self.X = None
        self.X_next = None
        self.diff = None
        self.state = None
        del self.diff_deque
        self.diff_deque = deque(maxlen=self.predstep+1)
        
        self.predictor.params_record['time'] = []
        self.predictor.params_record['alpha'] = []
        self.predictor.params_record['k'] = []
        
        if self.set_predY:
            rmse, mae, mape = self.get_error(np.asarray(self.set_predY), np.asarray(self.set_realY))
            print('TBSM: RMSE = {:.4f}, MAE = {:.4f}, MAPE = {:.4f}'.format(rmse, mae, mape))
            rmse, mae, mape = self.get_error(np.asarray(self.set_predY_), np.asarray(self.set_realY))
            print('ITBSM: RMSE = {:.4f}, MAE = {:.4f}, MAPE = {:.4f}'.format(rmse, mae, mape))

        self.set_predY = []
        self.set_predY_ = []
        self.set_realY = []
        
        if data_type == 'vali':
            self.pointer = random.randint(0, len(self.testX)-200-self.predstep) # the position of sample in test dataset.
            print('Env has been reset at %d'%(self.pointer))
        elif data_type == 'test':
            self.pointer = 0
        
        i = 0
        while i <= self.predstep:
            if data_type == 'vali':
                self.X = self.valiX[self.pointer]
                self.X_next = self.valiX[self.pointer+1]
                realY = self.valiY_nofilt[self.pointer]
            elif data_type == 'test':
                self.X = self.testX[self.pointer]
                self.X_next = self.testX[self.pointer+1]
                realY = self.testY_nofilt[self.pointer]
              
            predY, predY_ = self.predictor.predict(self.X)
            self.predictor.params_record['time'].append(self.pointer)
            
            self.set_predY.append(predY * self.maxv)
            self.set_predY_.append(predY_ * self.maxv)
            self.set_realY.append(realY * self.maxv)
            
            diff = (np.asarray(predY_) - np.asarray(realY)).reshape(self.predictor.link,)
            # print('self.diff shape (in ResetFunc):',self.diff.shape)
            self.diff_deque.append(diff)
            self.diff = self.diff_deque[0]
            
            self.pointer += 1
            i += 1
                    
        self.state = (self.X_next, self.diff)
        
        return self.state
    
    def render(self, mode='human'):
        return None
    
    def close(self):
        return None