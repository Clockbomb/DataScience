import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import cross_val_score


# Exercise 2.7

path = os.path.dirname(os.path.realpath(__file__)) + '/births.csv'
births =  pd.read_csv(path)

#print(births.head())

#Not sure why would this is a hint in c)
print(births['etnicity'].unique())

home_list =[]
pari_list=[]
etni_list=[]
for _ in range(len(births.index)):
    if(births.loc[_].at['child_birth'] == 'first line child birth, at home'):
        home_list.append('at_home')
    else:
        home_list.append('not_at_home')
    
    if(births.loc[_].at['parity'] == 1):
        pari_list.append('primi')
    else:
        pari_list.append('multi')

    if(births.loc[_].at['etnicity'] == 'Dutch'):
        etni_list.append('Dutch')
    else:
        etni_list.append('Not Dutch')


births.insert(loc=births.columns.get_loc('child_birth'), column='home', value= home_list)
births.drop(labels=['child_birth'], axis=1, inplace=True)

births.insert(loc=births.columns.get_loc('parity'), column='pari', value= pari_list)
births.drop(labels=['parity'], axis=1, inplace=True)

births.insert(loc=births.columns.get_loc('etnicity'), column='etni', value= etni_list)
births.drop(labels=['etnicity'], axis=1, inplace=True)


# d)
# useful link: https://towardsdatascience.com/python-scikit-learn-logistic-regression-classification-eb9c8de8938d

# Encoding of the categorical variables
urban_values = pd.get_dummies(births.urban, prefix='urban')
births = births.join(urban_values); births.drop(['urban'], axis=1, inplace=True)

age_cat_values = pd.get_dummies(births.age_cat, prefix='age_cat')
births = births.join(age_cat_values); births.drop(['age_cat'], axis=1, inplace=True)

births.etni = births.etni.map({'Not Dutch': 0, 'Dutch': 1})
births.pari = births.pari.map({'multi': 0, 'primi': 1})
births.home = births.home.map({'not_at_home': 0, 'at_home': 1})

# Target values
y = births.home.copy()
# Features 
x = births.drop(['provmin','age','home'],axis=1)

#print(births.head())
print(x.head())

# Implementing the model
model_1 = LogisticRegression().fit(x,y)

# Analysing the outcomes of the model
y_pred = pd.Series(model_1.predict(x))
target_names = ['not_at_home','at_home']
print(classification_report(y, y_pred, target_names = target_names))

# e)
model_2 = tree.DecisionTreeRegressor().fit(x,y)
tree.plot_tree(model_2)
plt.show()

# f)
# Performing 10-fold cross validation
scores_1 = cross_val_score(model_1, x, y, cv=10)
scores_2 = cross_val_score(model_2, x, y, cv=10)

print("The average accuracy for the Logistic Regression model is %0.2f, with %0.2f standard deviation."
    %(scores_1.mean(),scores_1.std()))

print("\nThe average accuracy for the Decision tree model is %0.2f, with %0.2f standard deviation."
    %(scores_2.mean(),scores_2.std()))

print("\nHence the Logistic Regression model fits better with the data!")