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

#zmiana daty na sam rok(format z date na int)

class Data():
    def __init__(self):
        self.df = pd.read_excel('main_roboczy.xlsx', sheet_name='Arkusz2')
        self.X = None
        self.Y = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def prep_data(self):
        self.df['year']= self.df['Zmienna'].dt.year
        self.df = self.df.drop('Zmienna', axis=1)
        self.df.set_index('year', inplace=True)
        #return self.df

    def print_correlation(self):
        sns.clustermap(self.df.corr())
        plt.show()

    def train_data(self):
        self.X = self.df.drop('Ludnosc', axis=1)
        self.Y = self.df['Ludnosc']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=12)

    def linear_reggresion(self):
        lnreg_model = LinearRegression()
        lnreg_model.fit(self.X_train, self.Y_train)
        self.Y_pred_lnreg = lnreg_model.predict(self.X_test)
        print(r2_score(self.Y_test,self.Y_pred_lnreg))

        print('MAE: ', mean_absolute_error(self.Y_test,self.Y_pred_lnreg))
        print('MSE: ', mean_squared_error(self.Y_test,self.Y_pred_lnreg))

        plt.figure(figsize=(8, 4))
        plt.plot(self.df.index, self.df['Ludnosc'], label='Dane historyczne', marker='o')
        plt.plot(self.X_test.index, self.Y_pred_lnreg, 'o',label='Prognoza')
        plt.xlabel('Rok')
        plt.ylabel('Liczba ludności')
        plt.legend()
        plt.show()


    def lgbm_regressor_model(self):
        lgb_model = lgb.LGBMRegressor(random_state=12)
        lgb_model.fit(self.X_train, self.Y_train)
        self.Y_pred_lgb = lgb_model.predict(self.X_test)
        print(r2_score(self.Y_test,self.Y_pred_lgb))

        print('MAE: ', mean_absolute_error(self.Y_test,self.Y_pred_lgb))
        print('MSE: ', mean_squared_error(self.Y_test,self.Y_pred_lgb))

        plt.figure(figsize=(8, 4))
        plt.plot(self.df.index, self.df['Ludnosc'], label='Dane historyczne', marker='o')
        plt.plot(self.X_test.index, self.Y_pred_lgb, 'o',label='Prognoza')
        plt.xlabel('Rok')
        plt.ylabel('Liczba ludności')
        plt.legend()
        plt.show()

    def random_forest_regressor_model(self):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=12)
        rf_model.fit(self.X_train, self.Y_train)   
        self.Y_pred_rand = rf_model.predict(self.X_test) 
        print(r2_score(self.Y_test,self.Y_pred_rand))
        print('MAE: ', mean_absolute_error(self.Y_test,self.Y_pred_rand,))
        print('MSE: ', mean_squared_error(self.Y_test,self.Y_pred_rand,))

        plt.figure(figsize=(8, 4))
        plt.plot(self.df.index, self.df['Ludnosc'], label='Dane historyczne', marker='o')
        plt.plot(self.X_test.index, self.Y_pred_rand, 'o',label='Prognoza')
        plt.xlabel('Rok')
        plt.ylabel('Liczba ludności')
        plt.legend()
        plt.show()
       

    def print_data(self):
        print(self.df.head())

d = Data()
d.prep_data()
d.train_data()
d.random_forest_regressor_model()
d.linear_reggresion()
d.lgbm_regressor_model()





