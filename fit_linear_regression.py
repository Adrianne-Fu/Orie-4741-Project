import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr',False)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

All = pd.read_csv(r'C:\Users\lyuli\Desktop\Orie-4741-Project\data\All.csv')
All['Offset'] = 1
data = All.drop(['Metro','Region Code','ZHVI_prev'], axis=1).dropna(axis=0, how='any').reset_index(drop = True) # tune this line to add/drop features
data = data[data.Date>'2010-01-01']
data.drop(columns=(['Date']), inplace=True)
print('Whole dataset display:\n',data.head())

# Feature scaling through standardization on non-categorical columns
category_feature = ['County','State']
data_category = data.loc[:, data.columns.isin(category_feature)]
data_int = data.loc[:, ~data.columns.isin(category_feature)]

std_columns = []
std_scaler = StandardScaler() # create a scaler object
data_int_std = pd.DataFrame(std_scaler.fit_transform(data_int), columns = data_int.columns)
standardized_data = pd.concat([data_int_std, data_category], axis = 1) # standardized dataset

data = pd.get_dummies(standardized_data)
# print(data)
# print(data.mean())
# print(data.std())



# split data to train-test
train_proportion = 0.8
n = len(data)
t = int(train_proportion * n)
target = data['ZHVI'] # response variable
data = data.loc[:, ~data.columns.isin(['ZHVI'])]
print(data.head())
train_x = np.array(data.iloc[:t,:])
test_x = np.array(data.iloc[t:,:])
train_y = np.array(target[:t])
test_y = np.array(target[t:])
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)

# This function computes the mean squared error
def MSE(y, pred):
    return np.mean( (np.array(y) - np.array(pred)) ** 2 ) # YOUR CODE HERE

# This function plots the main diagonal;for a "predicted vs true" plot with perfect predictions, all data lies on this line
def plotDiagonal(xmin, xmax):
    xsamples = np.arange(xmin,xmax,step=0.01)
    plt.plot(xsamples,xsamples,c='black')

# This helper function plots x vs y and labels the axes
def plotdata(x=None,y=None,xname=None,yname=None,margin=0.05,plotDiag=True,zeromin=False):
    plt.scatter(x,y,label='data')
    plt.xlabel(xname)
    plt.ylabel(yname)
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    if plotDiag:
        plotDiagonal(min(x)-margin*range_x,max(x)+margin*range_x)
    if zeromin:
        plt.xlim(0.0,max(x)+margin*range_x)
        plt.ylim(0.0,max(y)+margin*range_y)
    else:
        plt.xlim(min(x)-margin*range_x,max(x)+margin*range_x)
        plt.ylim(min(y)-margin*range_y,max(y)+margin*range_y)
    plt.show()

# This function plots the predicted labels vs the actual labels (We only plot the first 1000 points to avoid slow plots)
def plot_pred_true(test_pred=None, test_y=None, max_points = 1000):
    plotdata(test_pred[1:max_points], test_y[1:max_points],'Predicted', 'True', zeromin=True)


def run_OLS(train_y, test_y, train_vals, test_vals):

    # model fitting
    ols_model = sm.regression.linear_model.OLS(train_y.astype(float), train_vals.astype(float))
    while True:  # Bypasses SVD convergence assertion error
        try:
            results = ols_model.fit()
            break
        except:
            None

    w = np.array(results.params).reshape([len(results.params), 1])
    w_ = list(w.reshape((len(w),)))
    w_print = [format(i, 'f') for i in w_]
    print(w_print)

    train_pred = np.matmul(train_vals, w)
    test_pred = np.matmul(test_vals, w)

    train_MSE = MSE(train_y, train_pred.flatten())
    test_MSE = MSE(test_y, test_pred.flatten())

    return train_MSE, test_MSE, test_pred
train_MSE, test_MSE, test_pred = run_OLS(train_y, test_y, train_x, test_x)
print("Train MSE\t", str(train_MSE))
print("Test MSE\t", str(test_MSE))
plot_pred_true(test_pred.flatten(), test_y) #

