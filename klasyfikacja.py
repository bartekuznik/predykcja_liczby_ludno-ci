import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
sns.set()

class Data():
    def __init__(self):
        self.df = pd.read_excel('nowy.xlsx', sheet_name='Arkusz1')
        self.rslt_df = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None   
        self.best_res = -1
        self.best_i = 0

    def prep_data_class(self):
        sns.pairplot(self.df,hue='Region')

    
    def prep_data(self):
        self.X = self.df.drop('Region', axis=1)
        self.Y = self.df['Region']
        #return self.df  

    def random_tree(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=24)
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(self.X_train, self.Y_train)
        y_pred = classifier.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, y_pred)
        classification_rep = classification_report(self.Y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_rep)
        

d = Data()
d.prep_data()
d.random_tree()