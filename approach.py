# -*- coding: utf-8 -*-str()
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd
import numpy as np
year = 1881

year1 = pd.read_csv('yob1880.txt', names=['name', 'gender', 'frecuency'])
year1['year'] = np.zeros(len(year1))+1880
year2 = pd.read_csv('yob1881.txt', names=['name', 'gender', 'frecuency'])
year2['year'] = np.zeros(len(year2))+1881
df = pd.concat([year1, year2], axis = 'rows')

for i in range(0,107):
    year = year+1
    actual = pd.read_csv('yob'+str(year)+'.txt', names=['name', 'gender', 'frecuency'])
    actual['year'] = np.zeros(len(actual))+year
    
    df = pd.concat([df, actual], axis = 'rows')
    
df = df.groupby(['year', 'gender']).agg({'frecuency':'sum'}).reset_index()
df['gender'] = df['gender'].astype('category')
df = pd.get_dummies(df)

y = df['frecuency'].values
X = df.drop(['frecuency'], axis='columns').values

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X,y)

for i in range(1989,2017):
    year = i
    actual = pd.read_csv('yob'+str(year)+'.txt', names=['name', 'gender', 'frecuency'])
    actual['year'] = np.zeros(len(actual))+year
    
    df = pd.concat([df, actual], axis = 'rows')
    
df = df.groupby(['year', 'gender']).agg({'frecuency':'sum'}).reset_index()
df['gender'] = df['gender'].astype('category')
df = pd.get_dummies(df)

y_test = df['frecuency'].values
X_test = df.drop(['frecuency'], axis='columns').values

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: {0}'.format(rmse))

print('R squared: {0}'.format(model.score(X_test,y_test)))