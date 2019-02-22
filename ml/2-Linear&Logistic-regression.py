from sklearn.datasets import load_boston
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error
import numpy as np
boston = load_boston()
X_train,X_test,y_train,y_test = model_selection.train_test_split(boston.data,boston.target,test_size=0.20,random_state=0)

# # 线性回归
# regr = LinearRegression()
# regr.fit(X_train,y_train)
# y_pred = regr.predict(X_test)
# print('*'*20+"LinearRegression"+'*'*20)
# print("MAE: %0.3f\nMSE: %0.3f\nMSLE: %0.3f\nMEAE: %0.3f" % \
# 	(mean_absolute_error(y_test,y_pred),\
# 	mean_absolute_error(y_test,y_pred),\
# 	mean_squared_log_error(y_test,y_pred),\
# 	median_absolute_error(y_test,y_pred)))

# logistic回归
avg_price_house = np.average(boston.target)
high_price_idx = (y_train >= avg_price_house)
y_train[high_price_idx] = 1
y_train[np.logical_not(high_price_idx)] = 0
y_train = y_train.astype(np.int8)
high_price_idx = (y_test >= avg_price_house)
y_test[high_price_idx] = 1
y_test[np.logical_not(high_price_idx)] = 0
y_test = y_test.astype(np.int8)
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))