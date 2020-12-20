import numpy as np
from sklearn.naive_bayes import GaussianNB
from util import load_mnist,runCV1
from random import shuffle,seed

#######Data preprocessing######
path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)      #학습데이터와 레이블

seed(0)
numbers=list(range(len(d_mn)))
shuffle(numbers)

clf=GaussianNB()
clf=clf.fit(d_mn,l_mn)
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
results=runCV1(clf,shuffled_data,shuffled_labels)
print(np.mean(results)) #0.55

