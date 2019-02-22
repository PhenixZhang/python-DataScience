from sklearn import datasets
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
iris = datasets.load_iris()
X_train,X_test,y_train,y_test = model_selection.train_test_split(iris.data,iris.target,test_size=0.20,random_state=0)
clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))