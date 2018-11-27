# multi-label classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
iris = datasets.load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(iris.data,iris.target,test_size=0.50,random_state=4)
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(iris.target_names)
cm = confusion_matrix(Y_test,Y_pred)
#print(cm)
print('Accuracy:',metrics.accuracy_score(Y_test,Y_pred))
print('Precision:',metrics.precision_score(Y_test,Y_pred,average='weighted'))
print("recall:",metrics.recall_score(Y_test,Y_pred,average='weighted'))
print("F1 score:",metrics.f1_score(Y_test,Y_pred,average='weighted'))
img = plt.matshow(cm,cmap=plt.cm.autumn)
plt.colorbar(img,fraction=0.045)
for x in range(cm.shape[0]):
	for y in range(cm.shape[1]):
		plt.text(x,y,"%0.2f" % cm[x,y],size=12,color='black',ha='center',va='center')

plt.show()
print(classification_report(Y_test,Y_pred,target_names=iris.target_names))
