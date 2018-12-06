from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import model_selection
import numpy as np

if __name__ == '__main__':
	digits = load_digits()
	X = digits.data
	y = digits.target
	h1 = svm.LinearSVC(C=1.0)
	h2 = svm.SVC(kernel='rbf',degree=3,gamma=0.001,C=1.0)
	h3 = svm.SVC(kernel='poly',degree=3,C=1.0)

	chosen_random_state = 1
	cv_folds = 10
	eval_scoring = 'accuracy'
	workers = -1
	X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.30,random_state=chosen_random_state)
	kfolding = model_selection.KFold(n_splits=10,shuffle=True,random_state=1)
	for hypothesis in [h1,h2,h3]:
		scores = model_selection.cross_val_score(hypothesis,X_train,y_train,cv=cv_folds,scoring=eval_scoring,n_jobs=workers)
		print("%s -> cross validation accuracy: mean = %0.3f std = %0.3f" % (hypothesis,np.mean(scores),np.std(scores)))
		for train_index, validation_index in kfolding.split(X_train,y_train):
			hypothesis.fit(X[train_index],y[train_index])
			print(hypothesis.score(X[validation_index],y[validation_index]))
