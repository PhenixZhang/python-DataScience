import pickle
import urllib
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_20newsgroups
import requests

mnist = fetch_mldata("MNIST original")
pickle.dump(mnist,open("mnist.pickle","wb"))
target_page = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2"
with requests.get(target_page) as response:
	with open('ijcnn1.bz2','wb') as W:
		W.write(response.text)
target_page = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata"
cadata = load_svmlight_file(requests.get(target_page))
pickle.dump(cadata,open("cadata.pickle","wb"))
covertype_dataset = fetch_covtype(random_state=101,shuffle=True)
pickle.dump(covertype_dataset,open("covertype_dataset.pickle","wb"))
newsgroups_dataset = fetch_20newsgroups(shuffle=True,remove=('headers','footers','quotes'),random_state=6)
pickle.dump(newsgroups_dataset,open("newsgroups_dataset.pickle","wb"))
