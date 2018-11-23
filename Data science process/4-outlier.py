# # z-scorea
# from sklearn.datasets import load_boston
# boston = load_boston()
# continuous_variables = [n for n in range(boston.data.shape[1]) if n!=3]
# # print(continuous_variables)
# import numpy as np
# from sklearn import preprocessing
# normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:,continuous_variables])
# outliers_rows,outliers_columns = np.where(np.abs(normalized_data)>3)
# a = list(zip(outliers_rows,outliers_columns))
# # print('outliers_rows=',outliers_rows,'\noutliers_columns=',outliers_columns)
# print(a)
# for i in a:
# 	print(normalized_data[a

# # EllipticEnvelope
# import numpy as np
# from sklearn.datasets import make_blobs
# from sklearn.covariance import EllipticEnvelope
# from matplotlib import pyplot as plt
# blobs = 1
# blob = make_blobs(n_samples=100,n_features=2,centers=blobs,cluster_std=1.5,shuffle=True,random_state=5)
# robust_covariance_est = EllipticEnvelope(contamination=.1).fit(blob[0])
# detection = robust_covariance_est.predict(blob[0])
# outliers = np.where(detection==-1)[0]
# inliers = np.where(detection==1)[0]
# plt.scatter(blob[0][:,0],blob[0][:,1],c='blue',alpha=0.8,s=60,marker='o',edgecolors='white')
# plt.show()

# OneClassSVM
from sklearn.datasets import load_boston
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
from matplotlib import pyplot as plt
boston = load_boston()
continuous_variables = [n for n in range(boston.data.shape[1]) if n!=3]
normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:,continuous_variables])
pca = PCA(n_components=5)
Zscore_components = pca.fit_transform(normalized_data)
vtot = 'PCA Variance explained' + str(round(np.sum(pca.explained_variance_ratio_),3))
outliers_fraciton = 0.02
nu_estimate = 0.95 * outliers_fraciton + 0.05
machine_learning = svm.OneClassSVM(kernel='rbf',gamma=1.0/len(normalized_data),degree=3,nu=nu_estimate)
machine_learning.fit(normalized_data)
detection = machine_learning.predict(normalized_data)
outliers = np.where(detection==-1)
regular = np.where(detection==1)
for r in range(1,5):
	in_points = plt.scatter(Zscore_components[regular,0],Zscore_components[regular,r],c='blue',alpha=0.8,s=60,marker='o',edgecolors='white')
	out_points = plt.scatter(Zscore_components[outliers,0],Zscore_components[outliers,r],c='red',alpha=0.8,s=60,marker='o',edgecolors='white')
	plt.legend((in_points,out_points),('inliers','outliers'),scatterpoints=1,loc='best')
	plt.xlabel('Components1(' + str(round(pca.explained_variance_ratio_[0],3)) + ')')
	plt.ylabel('Components' + str(r+1) + '(' + str(round(pca.explained_variance_ratio_[r],3)) + ')')
	plt.xlim([-7,7])
	plt.ylim([-6,6])
	plt.title(vtot)
	plt.show()
