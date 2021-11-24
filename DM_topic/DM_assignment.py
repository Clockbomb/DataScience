import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set_theme(color_codes=True)
import statsmodels.api as sm


#path = __file__ + '/voorbeeld7_1.sav'

data_frame =  pd.read_spss('D:\\DataScience_Project\\DM_topic\\voorbeeld7_1.sav')

#print(data['chol'])

graph = sns.lmplot(x ='leeftijd', y ='chol', data = data_frame, fit_reg=True)
plt.show()


model = sm.formula.api.ols(formula = "chol ~ leeftijd",data = data_frame)¶
fit1 = model.fit()
fit1.summary()