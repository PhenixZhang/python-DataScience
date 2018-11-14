from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()

# # cov
# cov_data = np.corrcoef(iris.data.T)
# print(iris.feature_names)
# print(cov_data)
# img = plt.matshow(cov_data,cmap=plt.cm.rainbow)
# plt.colorbar(img,ticks=[-1,0,1],fraction=0.045)
# for x in range(cov_data.shape[0]):
# 	for y in range(cov_data.shape[1]):
# 		plt.text(x,y,"%0.2f" % cov_data[x,y],size=12,color='black',ha='center',va='center')
# plt.show()

# # PCA
# pca_2c = PCA(n_components=2)
# X_pca_2c = pca_2c.fit_transform(iris.data)
# print(X_pca_2c.shape)
# plt.scatter(X_pca_2c[:,0],X_pca_2c[:,1],c=iris.target,alpha=0.8,s=60,marker='o',edgecolors='white')
# print(pca_2c.explained_variance_ratio_.sum())
# plt.show()

# # Randomized -PCA
# rpca_2c = RandomizedPCA(n_components=2)
# X_rpca_2c = rpca_2c.fit_transform(iris.data)
# plt.scatter(X_rpca_2c[:,0],X_rpca_2c[:,1],c=iris.target,alpha=0.8,s=60,marker='o',edgecolors='white')
# print(rpca_2c.explained_variance_ratio_.sum())
# plt.show()

# # LFA
# fact_2c = FactorAnalysis(n_components=2)
# X_factor = fact_2c.fit_transform(iris.data)
# plt.scatter(X_factor[:,0],X_factor[:,1],c=iris.target,alpha=0.8,s=60,marker='o',edgecolors='white')
# plt.show()

# # LDA
# lda_2c = LDA(n_components=2)
# X_lda_2c = lda_2c.fit_transform(iris.data,iris.target)
# plt.scatter(X_lda_2c[:,0],X_lda_2c[:,1],c=iris.target,alpha=0.8,edgecolors='none')
# plt.show()

# # LSA
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# categories = ['sci.med','sci.space']
# twenty_sci_news = fetch_20newsgroups(categories=categories)
# tf_vect = TfidfVectorizer()
# word_freq = tf_vect.fit_transform(twenty_sci_news.data)
# tsvd_2c = TruncatedSVD(n_components=50)
# tsvd_2c.fit(word_freq)
# a = np.array(tf_vect.get_feature_names())[tsvd_2c.components_[20].argsort()[-10:][::-1]]
# print(a)

# # Kernel PCA
# from sklearn.decomposition import KernelPCA
# def circular_points(radius,N):
# 	return np.array([[np.cos(2*np.pi*t/N)*radius,np.sin(2*np.pi*t/N)*radius] for t in range(N)])
# N_points = 50
# fake_circular_data = np.vstack([circular_points(1.0,N_points),circular_points(5.0,N_points)])
# fake_circular_data += np.random.rand(*fake_circular_data.shape)
# fake_circular_target = np.array([0]*N_points +[1]*N_points)
# # plt.scatter(fake_circular_data[:,0],fake_circular_data[:,1],c=fake_circular_target,alpha=0.8,s=60,marker='o',edgecolors='white')
# kpca_2c = KernelPCA(n_components=2,kernel='rbf')
# X_kpca_2c = kpca_2c.fit_transform(fake_circular_data)
# plt.scatter(X_kpca_2c[:,0],X_kpca_2c[:,1],c=fake_circular_target,alpha=0.8,s=60,marker='o',edgecolors='white')
# plt.show()

# # T-SNE
# from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
# iris = load_iris()
# X,y = iris.data,iris.target
# X_tsne = TSNE(n_components=2).fit_transform(X)
# plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y,alpha=0.8,s=60,marker='o',edgecolors='white')
# plt.show()

# RBM
from sklearn import preprocessing
from sklearn.neural_network import BernoulliRBM
n_components = 64
olivetti_faces = datasets.fetch_olivetti_faces()
X = preprocessing.binarize(preprocessing.scale(olivetti_faces.data.astype(float)),0.5)
rbm = BernoulliRBM(n_components=n_components,learning_rate=0.01,n_iter=100)
rbm.fit(X)
plt.figure(figsize=(4.2,4))
for i,comp in enumerate(rbm.components_):
	plt.subplot(int(np.sqrt(n_components+1)),int(np.sqrt(n_components+1)),i+1)
	plt.imshow(comp.reshape((64,64)),cmap=plt.cm.gray_r,interpolation='nearest')
	plt.xticks(());plt.yticks(())
plt.suptitle(str(n_components) + 'components extracted by RBM',fontsize=16)
plt.subplots_adjust(0.08,0.02,0.92,0.85,0.08,0.23)
plt.show()
