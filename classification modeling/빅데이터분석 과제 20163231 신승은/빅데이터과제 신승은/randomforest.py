#randomforest 0.96
from sklearn.ensemble import RandomForestClassifier
from util import runCV2,load_mnist
from random import shuffle,seed
import time
import numpy as np


path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)

seed(0)
numbers=list(range(len(d_mn)))
shuffle(numbers)
clf=RandomForestClassifier()
shuffled_data=d_mn[numbers]
shuffeld_labels=l_mn[numbers]
results=runCV2(clf,shuffled_data,shuffeld_labels,isAcc=True)
print(np.mean(results))#0.96