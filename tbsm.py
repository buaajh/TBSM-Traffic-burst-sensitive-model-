# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd
import math
import random
from preprocess import DataPreprocess as dp
import warnings

warnings.filterwarnings('ignore')

class TBSM():
    def __init__(self):
        self.link = 257
        
        self.dp = dp()
        self.inputstep = self.dp.inputstep
        
        if os.path.exists('data%dp%d/trainX.h5'%(self.dp.inputstep,self.dp.predstep)) and os.path.exists('data%dp%d/trainY.h5'%(self.dp.inputstep,self.dp.predstep)) and os.path.exists('data%dp%d/trainY_nofilt.h5'%(self.dp.inputstep,self.dp.predstep)):
            print('loading train data from the local...')
            with h5py.File('data%dp%d/trainX.h5'%(self.dp.inputstep,self.dp.predstep)) as f:
                self.trainX = f['data'][:]
            with h5py.File('data%dp%d/trainY.h5'%(self.dp.inputstep,self.dp.predstep)) as f:
                self.trainY = f['data'][:]
            with h5py.File('data%dp%d/trainY_nofilt.h5'%(self.dp.inputstep,self.dp.predstep)) as f:
                self.trainY_nofilt = f['data'][:]
        else:
            self.trainX, self.trainY, self.trainY_nofilt = self.dp.get_data(data_type='train')
            
            if not os.path.exists('data%dp%d'%(self.dp.inputstep,self.dp.predstep)):
                os.makedirs('data%dp%d'%(self.dp.inputstep,self.dp.predstep))
            
            with h5py.File('data%dp%d/trainX.h5'%(self.dp.inputstep,self.dp.predstep),'w') as f:
                f['data'] = self.trainX
            with h5py.File('data%dp%d/trainY.h5'%(self.dp.inputstep,self.dp.predstep),'w') as f:
                f['data'] = self.trainY
            with h5py.File('data%dp%d/trainY_nofilt.h5'%(self.dp.inputstep,self.dp.predstep),'w') as f:
                f['data'] = self.trainY_nofilt

        self.trainX_cl = self.delete_corr(self.trainX)
        self.trainX_incr = self.increment(self.trainX, self.trainY)
        self.params_record = {'time':[], 'alpha': [], 'k': []}
        
    def delete_corr(self, X):
        corr_link = pd.read_csv(r'E:/gongtiS/gisData/delcorrlink.csv')
        corr_link = list(np.asarray(corr_link)[:,1])
        if len(np.shape(X)) == 3:
            return X[:,:,corr_link]
        else:
            return X[:,corr_link]
    
    def trend(self, X):
        return X[-1] - X[0]
    
    def increment(self, X, Y):
        incr = []
        for i in range(len(X)):
            incr.append(Y[i]-X[i][-1])
        return incr
    
    def distance(self, X1, X2, dtype='ed'):
        """calcuate distance between X1 and X2.
        
        Arguments:
            X1, X2: the objects whose distance would be calcuated.
            dtype: the type of distacne, `ed` for Euclidean distance, or `cd` for consine distace.
            
        Returns:
            d: distance between X1 and X2.
        """
        X1, X2 = np.asarray(X1), np.asarray(X2)
        
        if dtype == 'ed': # Euclidean distance
            d = np.linalg.norm(X1-X2, ord=2)
        elif dtype == 'cd': # Cosine distance
            d = 1 - np.dot(X1, X2)/(np.linalg.norm(X1, ord=2)*np.linalg.norm(X2, ord=2))
        else:
            raise Exception('Wrong or missing argument of `dtype`, please use `ed` for Euclidean distance, or `cd` for consine distace.')
        return d
    
    
    def index(self, l):
        """create a index dict of list 'l' ().
        
        Arguments:
            l: a list.
            
        Returns:
            index: a dict, whose keys are elements of l, and values are indexes of l's elements.
        """
        index = {}
        for i in range(len(l)):
            index[l[i]] = i
        return index
    
    def quickSort(self, l):
        if len(l) < 2:
            return l
        else:
            pivot = l[0]
            left = [i for i in l[1:] if i <= pivot]
            right = [i for i in l[1:] if i >= pivot]
        return self.quickSort(left) + [pivot] + self.quickSort(right)
    
    def mergeSort(self, lst):
        #合并左右子序列函数
        def merge(arr,left,mid,right):
            temp=[]     #中间数组
            i=left          #左段子序列起始
            j=mid+1   #右段子序列起始
            while i<=mid and j<=right:
                if arr[i]<=arr[j]:
                    temp.append(arr[i])
                    i+=1
                else:
                    temp.append(arr[j])
                    j+=1
            while i<=mid:
                temp.append(arr[i])
                i+=1
            while j<=right:
                temp.append(arr[j])
                j+=1
            for i in range(left,right+1):    #  !注意这里，不能直接arr=temp,他俩大小都不一定一样
                arr[i]=temp[i-left]
        #递归调用归并排序
        def mSort(arr,left,right):
            if left>=right:
                return
            mid=(left+right)//2
            mSort(arr,left,mid)
            mSort(arr,mid+1,right)
            merge(arr,left,mid,right)
     
        n=len(lst)
        if n<=1:
            return lst
        mSort(lst,0,n-1)
        return lst
    
    
    
    def index_sorted(self, l):
        """return original index with a sorted order.
        
        Arguments:
            
        Returns:
            ind_sorted: the new index of sorted l
        """
        index_of_l = self.index(l)
        #sorted_l = self.quickSort(l)
        sorted_l = self.mergeSort(l)
        ind_sorted = []
        for element in sorted_l:
            ind_sorted.append(index_of_l[element])
        
        return ind_sorted
    
    
    def predict(self, predX, aparams=None):
        """use SEKNN to predict sample predX
        
        Arguments:
            #params: parameters, incluing (k, alpha), i.e. action in RL.
            predX: sample. (Attention: predX is a single sameple (one time step), not a dataset)
            
        Returns:
            pred: predicted valule of X.
        """
        params = {0:(0.9, 97), 1:(0.9, 57), 2:(0.8, 57), 3:(0.8, 54), 4:(0.7, 54)}
        alpha, k = params[self.dp.predstep]
        if aparams is None:
            alpha_, k_ = params[self.dp.predstep]
        else:
            alpha_ = aparams[0]
            k_ = max(20, int(aparams[1]*100))
        
        self.params_record['alpha'].append(alpha_)
        self.params_record['k'].append(k_)
        
        predX_cl = self.delete_corr(predX)
        
        ED = []
        CD = []
        for index_t, t in enumerate(self.trainX_cl):
            ED.append(self.distance(t[-1], predX_cl[-1], dtype='ed'))
            CD.append(self.distance(self.trend(t), self.trend(predX_cl), dtype='cd'))
        maxED, minED = max(ED), min(ED)
        ED01 = [2*(d-minED)/(maxED-minED) for d in ED]
        
        D = list(map(lambda ed, cd: alpha*ed + (1-alpha)*cd, ED01, CD))
        D_ = list(map(lambda ed, cd: alpha_*ed + (1-alpha_)*cd, ED01, CD))

        
        D_topK = self.index_sorted(D)[:k] # the index value
        D_topK_ = self.index_sorted(D_)[:k_]
        
        nn_topK = np.asarray(self.trainX_incr)[D_topK].reshape(-1, self.link)
        nn_topK_ = np.asarray(self.trainX_incr)[D_topK_].reshape(-1, self.link)
        
        w = [math.e**(-D[i]**2/(2*1.33**2)) for i in D_topK]
        w_ = [math.e**(-D_[i]**2/(2*1.33**2)) for i in D_topK_]
        
        predY = np.dot(w, nn_topK + predX[-1]) / np.sum(w)
        predY_ = np.dot(w_, nn_topK_ + predX[-1]) / np.sum(w_)
        return predY, predY_

    def evaluate(self, y, y_):
        rmse = np.sqrt(np.mean(np.square(y-y_)))
        mae = np.mean(np.abs(y-y_))
        mape = np.mean(np.abs(y-y_)/y)*100
        return rmse, mae, mape


## FUNCTIONAL TEST
#
#if __name__ == '__main__':
#    model = SEKNN()
#    for p in [4]:# range(5):
#        model.dp.predstep = p
#        testX, testY, testY_nofilt = model.dp.get_data(data_type='test')
#
#        y = np.array([i * model.dp.maxv for i in testY_nofilt]).reshape(-1, 257)
#        df = pd.DataFrame(y)
#        df.to_csv('model/testy-only-predY%dp%d.csv'%(model.dp.inputstep, model.dp.predstep), header=False, index=False)
#
#        set_predY = []
#        for x in testX:
#            set_predY.append(model.predict(x) * model.dp.maxv)
#
#        y_ = np.array(set_predY).reshape(-1, 257)
#        df = pd.DataFrame(y_)
#        df.to_csv('model/seknn-only-predY%dp%d.csv'%(model.dp.inputstep, model.dp.predstep), header=False, index=False)
#
#        rmse, mae, mape = model.evaluate(y, y_)
#        print('predstep={},RMSE = {:.4}, MAE = {:.4}, MAPE = {:.4}'.format(p, rmse, mae, mape))
