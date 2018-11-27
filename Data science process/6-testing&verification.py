from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import model_selection
digits = load_digits()
#print(digits.DESCR)
X = digits.data
y = digits.target
h1 = svm.LinearSVC(C=1.0)
h2 = svm.SVC(kernel='rbf',degree=3,gamma=0.001,C=1.0)
h3 = svm.SVC(kernel='poly',degree=3,C=1.0)
# h1.fit(X,y);h2.fit(X,y);h3.fit(X,y)
# print("h1_score={0}\nh2_score={1}\nh3_score={2}".format(h1.score(X,y),h2.score(X,y),h3.score(X,y)))

# chosen_random_state = 1
# X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.30,random_state=chosen_random_state)
# print("X train shape %s,X test shape %s,\ny train shape %s,y test shape %s"%(X_train.shape,X_test.shape,y_test.shape,y_train.shape))
# h1.fit(X_train,y_train)
# print(h1.score(X_test,y_test))

chosen_random_state = 1
X_train,X_validation_test,y_train,y_validation_test = model_selection.train_test_split(X,y,test_size=0.40,random_state=chosen_random_state)
X_validation,X_test,y_validation,y_test = model_selection.train_test_split(X_validation_test,y_validation_test,test_size=0.50,random_state=chosen_random_state)
# print("X train shape:%s,X validation shape:%s,X test shape:%s,\ny train shape:%s,y validation shape:%s,y test shape:%s\n"\
	# %(X_train.shape,X_validation.shape,X_test.shape,y_train.shape,y_validation.shape,y_test.shape))
for hypothesis in [h1,h2,h3]:
	hypothesis.fit(X_train,y_train)
	print("%s -> validation mean accuracy = %0.3f" % (hypothesis,hypothesis.score(X_validation,y_validation)))
	h2.fit(X_train,y_train)
	print("\n%s -> test mean accuracy = %0.3f" % (h2,h2.score(X_test,y_test)))
	print("*"*20)
