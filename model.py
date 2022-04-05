# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:05:54 2022

@author: amitava
"""
import numpy as np
import pandas as pd
import seaborn as sns 
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('ce_optimum.csv')
# y includes our labels and x includes our features
y = data.Process                         # M or B 
list = ['Process']
x = data.drop(list,axis = 1 )
x.head()
y
      # M = 212, B = 357

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)


classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
#from sklearn.svm import SVC
#sv = SVC(kernel='linear').fit(x_train,y_train)

import pickle
pickle.dump(classifier, open('hot_roll_model.pkl', 'wb'))