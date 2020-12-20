from sklearn.neural_network import MLPClassifier as mlp
from util import runCV2,load_mnist
from random import shuffle,seed
import time
import numpy as np


path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)

"""
clf=mlp(hidden_layer_sizes=100,max_iter=500,learning_rate_init=0)
clf=clf.fit(d_mn[:200],l_mn[:200])
pred=clf.predict(d_mn[200:400])
correct=pred==l_mn[200:400]
acc=sum(correct)/len(correct)
#learning rate를 올렸을때 성능이 낮아짐 0.76 그렇다고 안좋은 건 아님
"""
seed(0)
numbers=list(range(len(d_mn)))
shuffle(numbers)
clf=mlp(hidden_layer_sizes=20,max_iter=500)
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
st=time.time()
results=runCV2(clf,shuffled_data,shuffled_labels,isAcc=False)
et=time.time()-st
print(et)
print(np.mean(results))
