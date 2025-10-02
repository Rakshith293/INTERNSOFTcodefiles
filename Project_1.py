#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:26:51 2025

@author: madangowda
"""

#IMPORT THE LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt

#Read data from the files

data = pd.read_csv("advertising.csv")
data.head()

#Visulaize data

fig, axs = plt.subplots(1,3)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

#CREATING X&Y FOR LINEaR REGRESSION

feature_cols=['TV']
x=data[feature_cols]
y = data.Sales

#IMPORTING LINEAR REGRESSION ALGO
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


#y=a+bx

result=6.9748+0.0554*30
print(result)


#CREATE A DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE

X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()



preds=lr.predict(X_new)
preds

data.plot(kind='scatter',x='TV',y="Sales")
plt.plot(X_new,preds,c='green',linewidth=3)




import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales~TV',data=data).fit()
lm.conf_int()




#FINDING THE PROBABILITY VALUES
lm.pvalues



#FINDING THE RSQUARED VALUES
lm.rsquared

#MULTILINEAR REGRESSION
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

lr=LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()
