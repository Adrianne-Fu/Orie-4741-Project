import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option('expand_frame_repr',False)

data = pd.read_csv(r'C:\Users\lyuli\Desktop\Orie-4741-Project\data\All.csv')
data = data[data.Date>'2010-01-01']
data.drop(columns={'Date','Region Code','Metro'},inplace=True)

data.dropna(inplace=True)

X = pd.get_dummies(data).drop(columns={'ZHVI'})

y = data.ZHVI
print(X)
print(X.shape)
print(y.shape)
reg = LinearRegression(fit_intercept=True).fit(X, y)

w = reg.coef_
w = [format(i, 'f') for i in w]
print(w)

features = X.columns
for f in range(len(features)):
    print(features[f],w[f])