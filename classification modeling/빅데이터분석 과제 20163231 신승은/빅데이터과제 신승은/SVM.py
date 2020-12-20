from sklearn.svm import LinearSVC

import time
from random import shuffle,seed
from util import runCV1,load_mnist
import numpy as np

path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)

clf=LinearSVC()
numbers=list(range(len(d_mn)))

seed(0)
shuffle(numbers)
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
st=time.time()
results=runCV1(clf,shuffled_data,shuffled_labels)
lt=time.time()-st
print(np.mean(results))
