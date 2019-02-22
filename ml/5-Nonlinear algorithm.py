# # SVC
# from sklearn.datasets import load_svmlight_file
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC

# X_train,y_train = load_svmlight_file('ijcnn1.bz2')
# first_rows = 2500
# X_train,y_train = X_train[:first_rows,:],y_train[:first_rows]
# hypothesis = SVC(kernel='rbf',random_state=101)
# scores = cross_val_score(hypothesis,X_train,y_train,cv=5,scoring='accuracy')
# print("SVC with rbf kernel -> cross validation accuracy: mean = %0.3f std = %0.3f"%\
# 	(np.mean(scores),np.std(scores)))

# SVR
import pickle
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
X_train,y_train = pickle.load(open("cadata.pickle","rb"))
first_rows = 2000
X_train = scale(X_train[:first_rows,:].toarray())
y_train = y_train[:first_rows]/10**4.0
hypothesis = SVR()
scores = cross_val_score(hypothesis,X_train,y_train,cv=3,scoring='neg_mean_absolute_error')
print("SVR -> cross validation accuracy: mean = %0.3f std = %0.3f" %\
	(np.mean(scores),np.std(scores)))

# # SVR optimization
# from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.datasets import load_svmlight_file
# if __name__ == '__main__':
# 	X_train,y_train = load_svmlight_file('ijcnn1.bz2')
# 	first_rows = 2500
# 	X_train,y_train = X_train[:first_rows,:],y_train[:first_rows]
# 	hypothesis = SVC(kernel='rbf',random_state=101)
# 	search_dict = {'C':[0.01,0.1,1,10,100],'gamma':[0.1,0.01,0.001,0.0001]}
# 	search_func = RandomizedSearchCV(estimator=hypothesis,param_distributions=search_dict,n_iter=10,scoring='accuracy',n_jobs=-1,iid=True,refit=True,cv=5,random_state=101)
# 	search_func.fit(X_train,y_train)
# 	print('Best parameters %s' % search_func.best_params_)
# 	print('cross validation accuracy: mean = %0.3f' % search_func.best_score_)