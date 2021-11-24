import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set_theme(color_codes=True)
import statsmodels.api as sm
import os

# Exercise 2.6

path = os.path.dirname(os.path.realpath(__file__)) + '/voorbeeld7_1.sav'
chol1 =  pd.read_spss(path)


# a)
graph = sns.lmplot(x ='leeftijd', y ='chol', data = chol1, fit_reg=True)

# b)
model = sm.formula.ols(formula = "chol~leeftijd", data = chol1)
fit1 = model.fit()
print(fit1.summary())

# c)
fit2 = sm.formula.ols(formula = "chol ~ leeftijd + bmi + sekse + alcohol", data = chol1).fit()
#print(fit2.summary())
# Which factors are statiscally significant?

# d)
chol1['Residuals'] = fit2.resid

plt.figure()
sns.histplot(chol1['Residuals'])

plt.show()