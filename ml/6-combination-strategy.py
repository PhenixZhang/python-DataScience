# # bagging
# # import pickle
# # import numpy as np
# # from sklearn.model_selection import cross_val_score
# # from sklearn.ensemble import BaggingClassifier
# # from sklearn.neighbors import KNeighborsClassifier
# if __name__ == '__main__':
# 	covertype_dataset = pickle.load(open("covertype_dataset.pickle","rb"))
# 	# print(covertype_dataset.DESCR)
# 	covertype_X = covertype_dataset.data[:15000,:]
# 	covertype_y = covertype_dataset.target[:15000]
# 	covertypes = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
# 	hypothesis = BaggingClassifier(KNeighborsClassifier(n_neighbors=1),max_samples=0.7,max_features=0.7,n_estimators=100)
# 	# hypothesis = KNeighborsClassifier(n_neighbors=1)
# 	scores = cross_val_score(hypothesis,covertype_X,covertype_y,cv=3,scoring='accuracy',n_jobs=-1)
# 	print("BaggingClassifier -> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(scores),np.std(scores)))

# Random forest
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
if __name__ == '__main__':
	covertype_dataset = pickle.load(open("covertype_dataset.pickle","rb"))
	# print(covertype_dataset.DESCR)
	covertype_X = covertype_dataset.data[:15000,:]
	covertype_y = covertype_dataset.target[:15000]
	covertypes = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
	hypothesis = RandomForestClassifier(n_estimators=100,random_state=101)
	scores = cross_val_score(hypothesis,covertype_X,covertype_y,cv=3,scoring='accuracy',n_jobs=-1)
	print("ExtraTreesClassifier -> cross validation accuracy: mean = %0.3f std = %0.3f" % \
		(np.mean(scores),np.std(scores)))

# # CalibrateClassifierCV
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.calibration import CalibratedClassifierCV,calibration_curve
# hypothesis = RandomForestClassifier(n_estimators=100,random_state=101)
# calibration = CalibratedClassifierCV(hypothesis,method='sigmoid',cv=5)
# covertype_dataset = pickle.load(open("covertype_dataset.pickle","rb"))
# covertype_X = covertype_dataset.data[:15000,:]
# covertype_y = covertype_dataset.target[:15000]
# covertypes = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
# covertype_test_X = covertype_dataset.data[15000:25000,:]
# covertype_test_y  =covertype_dataset.target[15000:25000]
# hypothesis.fit(covertype_X,covertype_y)
# calibration.fit(covertype_X,covertype_y)
# prob_row = hypothesis.predict_proba(covertype_test_X)
# prob_cal = calibration.predict_proba(covertype_test_X)
# tree_kind = covertypes.index('Ponderosa Pine')
# probs = pd.DataFrame(list(zip(prob_row[:,tree_kind],prob_cal[:,tree_kind])),columns=['raw','calibrated'])
# plot = probs.plot(kind='scatter',x=0,y=1,s=64,c='blue',edgecolors='white')
# plt.show()

# # AdaBoost
# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.model_selection import cross_val_score
# if __name__ == '__main__':
# 	covertype_dataset = pickle.load(open("covertype_dataset.pickle","rb"))
# 	covertype_X = covertype_dataset.data[:15000,:]
# 	covertype_y = covertype_dataset.target[:15000]
# 	covertypes = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
# 	hypothesis = AdaBoostClassifier(n_estimators=300,random_state=101)
# 	scores = cross_val_score(hypothesis,covertype_X,covertype_y,cv=3,scoring='accuracy',n_jobs=-1)
# 	print("AdaBoostClassifier -> cross validation accuracy: mean = %0.3f std = %0.3f" %\
# 		(np.mean(scores),np.std(scores)))

# # Gradient Tree Boosting(GTB)
# import pickle
# import numpy as np
# from sklearn.model_selection import cross_val_score,StratifiedKFold
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# if __name__ == '__main__':
# 	covertype_dataset = pickle.load(open("covertype_dataset.pickle","rb"))
# 	covertype_X = covertype_dataset.data[:15000,:]
# 	covertype_y = covertype_dataset.target[:15000] - 1
# 	# covertypes = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
# 	covertype_validation_X = covertype_dataset.data[15000:20000,:]
# 	covertype_validation_y = covertype_dataset.target[15000:20000] - 1
# 	covertype_test_X = covertype_dataset.data[20000:25000,:]
# 	covertype_test_y = covertype_dataset.target[20000:25000] - 1
# 	# print(covertype_test_X.shape,covertype_test_y.shape)
# 	hypothesis = GradientBoostingClassifier(max_depth=5,n_estimators=50,random_state=101)
# 	hypothesis.fit(covertype_X,covertype_y)
# 	print("GradientBoostingClassifier -> cross validation accuracy: %0.3f" % \
# 		accuracy_score(covertype_test_y,hypothesis.predict(covertype_test_X)))

# # XGBoost
# import pickle
# from sklearn.model_selection import cross_val_score,StratifiedKFold
# import xgboost as xgb
# from sklearn.metrics import accuracy_score,confusion_matrix
# covertype_dataset = pickle.load(open("covertype_dataset.pickle","rb"))
# covertype_dataset.target = covertype_dataset.target.astype(int)
# covertype_X = covertype_dataset.data[:15000,:]
# covertype_y = covertype_dataset.target[:15000] - 1
# covertype_validation_X = covertype_dataset.data[15000:20000,:]
# covertype_validation_y = covertype_dataset.target[15000:20000] - 1
# covertype_test_X = covertype_dataset.data[20000:25000,:]
# covertype_test_y = covertype_dataset.target[20000:25000] - 1
# hypothesis = xgb.XGBClassifier(objective="multi:softprob",max_depth=24,gamma=0.1,subsample=0.90,learning_rate=0.01,n_estimators=500,nthread=-1)
# hypothesis.fit(covertype_X,covertype_y,eval_set=[(covertype_validation_X,covertype_validation_y)],eval_metric='merror',early_stopping_rounds=25,verbose=False)
# print("test accuracy:",accuracy_score(covertype_test_y,hypothesis.predict(covertype_test_X)))
# print(confusion_matrix(covertype_test_y,hypothesis.predict(covertype_test_X)))