from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from random import shuffle,seed
from util import runCV2,load_mnist
import numpy as np

path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)

numbers=list(range(len(d_mn)))
seed(0)
shuffle(numbers)
clf=LinearDiscriminantAnalysis()
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
st=time.time()
results=runCV2(clf,shuffled_data,shuffled_labels,isAcc=True)
et=time.time()-st
print(np.mean(results))


