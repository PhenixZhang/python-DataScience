from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import model_selection

if __name__ == '__main__':
	digits = load_digits()
	X,y = digits.data,digits.target
	h = svm.SVC()
	hp = svm.SVC(probability=True,random_state=1)
	search_grid = [{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}]
	scorer = 'accuracy'
	search_func = model_selection.GridSearchCV(estimator=h,param_grid=search_grid,scoring=scorer,n_jobs=-1,iid=False,refit=True,cv=10)
	search_func.fit(X,y)
	print(search_func.best_estimator_,search_func.best_params_,search_func.best_score_)
