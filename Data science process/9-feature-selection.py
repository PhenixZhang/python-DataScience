# # 基于方差的特征选择
# from sklearn.datasets import make_classification
# from sklearn.feature_selection import VarianceThreshold
# import numpy as np
# X,y = make_classification(n_samples=10,n_features=5,n_informative=3,n_redundant=0,random_state=101)
# print(X,y)
# print("variance:",np.var(X,axis=0))
# X_selected = VarianceThreshold(threshold=1.0).fit_transform(X)
# print("Before:",X[0,:])
# print("After:",X_selected[0,:])

# # 单变量选择
# from sklearn.datasets import make_classification
# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import chi2,f_classif
# from sklearn.preprocessing import Binarizer,scale
# import numpy as np
# X,y = make_classification(n_samples=800,n_features=100,n_informative=25,n_redundant=0,random_state=101)
# Xbin = Binarizer().fit_transform(scale(X))
# print(Xbin.shape)
# selector_chi2 = SelectPercentile(chi2,percentile=25).fit(Xbin,y)
# selector_f_classif = SelectPercentile(f_classif,percentile=25).fit(X,y)
# chi_scores = selector_chi2.get_support()
# f_classif_scores = selector_f_classif.get_support()
# selected = chi_scores & f_classif_scores
# print(type(selected))

# # 递归消除
# from sklearn import model_selection
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFECV #递归消除和交叉验证
# X,y = make_classification(n_samples=100,n_features=100,n_informative=5,n_redundant=2,random_state=101)
# X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.30,random_state=101)
# classifier = LogisticRegression(random_state=101)
# classifier.fit(X_train,y_train)
# print("In-sample accuracy: %0.3f" % classifier.score(X_train,y_train))
# print("Out-sample accuracy: %0.3f" % classifier.score(X_test,y_test))
# selector = RFECV(estimator=classifier,step=1,cv=10,scoring='accuracy')
# selector.fit(X_train,y_train)
# print("Optimal number of feaures: %d" % selector.n_features_)
# X_train_s = selector.transform(X_train)
# X_test_s = selector.transform(X_test)
# classifier.fit(X_train_s,y_train)
# print("Out-sample accuracy: %0.3f" % classifier.score(X_test_s,y_test))

# Lasso
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.datasets import make_classification
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import make_regression
X,y = make_classification(n_samples=100,n_features=100,n_informative=5,n_redundant=2,random_state=101)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.30,random_state=101)
classifier = LogisticRegression(C=0.1,penalty='l1',random_state=101)
classifier.fit(X_train,y_train)
print("Out-of-sample accuracy: %0.3f" % classifier.score(X_test,y_test))
selector = RandomizedLogisticRegression(n_resampling=300,random_state=101)
selector.fit(X_train,y_train)
print("Variance selected: %i" % sum(selector._get_support_mask()!=0))
X_train_s = selector.transform(X_train) 
X_test_s = selector.transform(X_test)
classifier.fit(X_train_s,y_train)
print("Out-of-sample accuracy: %0.3f" % classifier.score(X_test_s,y_test))
XX,yy = make_regression(n_samples=100,n_features=10,n_informative=4,random_state=101)
rlasso = RandomizedLasso()
rlasso.fit(XX,yy)
print(list(enumerate(rlasso.scores_)))
