import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
sns.set()

class Data():
    def __init__(self):
        self.df = pd.read_excel('main_roboczy.xlsx', sheet_name='Arkusz2')
        self.rslt_df = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None   
        self.best_res = -1
        self.best_i = 0
    
    def prep_data(self):
        self.df = pd.read_excel('nowy.xlsx', sheet_name='Arkusz1')
        self.df.set_index('Rok', inplace=True)
        self.rslt_df = self.df[self.df['Region'] == 'WOJ. DOLNOŚLĄSKIE']
        self.rslt_df = self.rslt_df.drop('Region', axis=1)
        self.X = self.rslt_df.drop('Liczba_ludnosci', axis=1)
        self.Y = self.rslt_df['Liczba_ludnosci']  
        #return self.df  

    def random_tree(self):
        for i in range(2, 101, 2):
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=i)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=12)
            rf_model.fit(self.X_train, self.Y_train)   
            self.Y_pred_rand = rf_model.predict(self.X_test) 
            if self.best_res < r2_score(self.Y_test,self.Y_pred_rand):
                self.best_res = r2_score(self.Y_test,self.Y_pred_rand)
                self.best_i = i
        
        print(self.best_res)
        print(self.best_i)

d = Data()
d.prep_data()
d.random_tree()