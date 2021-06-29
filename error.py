# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

#scenario = 'SE'
#scenario = 'Non-SE'
scenario = 'All'
print('********* Current scenario is %s. **********'%scenario)


for i in [0]:#range(5):
    print('######### Current predstep is %d. #########'%i)
    link = 257
    #predY_all = pd.read_csv(r'E:/gongtiS/result/SAE/pred5p2.csv').as_matrix()[:,1:]
    predY_all = pd.read_csv(r'H:/AutoKNN/AutoKNN/adaptive-model/model2p0/1000-reinforced-predY2p%d.csv'%i,header=None).values
    testY_all = pd.read_csv(r'H:/AutoKNN/AutoKNN/adaptive-model/model2p0/1000-testY2p%d.csv'%i,header=None).as_matrix()[1:]
    
    if scenario == 'SE':
        """
        SE
        """
        predY = np.vstack((predY_all[180:691,:],predY_all[900:1411,:]))
        testY = np.vstack((testY_all[180:691,:],testY_all[900:1411,:]))
    elif scenario == 'Non-SE':
        """
        Non-SE
        """
        predY = np.vstack((predY_all[:180,:],predY_all[691:900,:],predY_all[1411:,:]))
        testY = np.vstack((testY_all[:180,:],testY_all[691:900,:],testY_all[1411:,:]))
    elif scenario == 'All':
        """
        SE
        """
        predY = predY_all
        testY = testY_all

    
    
    RMSE = np.sqrt(np.mean(np.square(np.asarray(predY).reshape(-1,link)-np.asarray(testY).reshape(-1,link))))
    MAE = np.mean(np.abs(np.asarray(testY).reshape(-1,link) - np.asarray(predY).reshape(-1,link)))
    MAPE = np.mean(np.abs(np.asarray(predY).reshape(-1,link)-np.asarray(testY).reshape(-1,link))/np.asarray(testY).reshape(-1,link))
    print('RMSE:%.4f'%RMSE)
    print('MAE:%.4f'%MAE)
    print('MAPE:%.4f%%'%(MAPE*100))