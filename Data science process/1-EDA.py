import pandas as pd
import matplotlib.pyplot as plt
iris_filename = 'datasets-uci-iris.csv'
iris = pd.read_csv(iris_filename,header = None,names=['sepal_length','sepal_width','petal_length','petal_width','target'])
print(iris.head())
print(iris.describe())
boxes = iris.boxplot(return_type='axes')
print(iris.quantile([0.1,0.5,0.9]))
print(type(iris.target.unique()))
print(pd.crosstab(iris['petal_length'] > 3.758667,iris['petal_width']>1.198667))
scatterplot = iris.plot(kind='scatter',x='petal_width',y='petal_length',s=64,c='blue',edgecolors='white')
distr = iris.petal_width.plot(kind='hist',alpha=0.5,bins=20)
plt.show()
