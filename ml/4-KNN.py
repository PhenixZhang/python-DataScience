import pickle
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

mnist = pickle.load(open("mnist.pickle","rb"))
mnist.data,mnist.target = shuffle(mnist.data,mnist.target)
mnist.data = mnist.data
mnist.target = mnist.target
X_train,X_test,y_train,y_test = model_selection.train_test_split(mnist.data,mnist.target,test_size=0.80,random_state=0)
clf = KNeighborsClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))
