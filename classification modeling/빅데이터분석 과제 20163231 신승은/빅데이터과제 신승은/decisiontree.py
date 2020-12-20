import numpy as np
from sklearn.tree import DecisionTreeClassifier
from util import load_mnist

#######Data preprocessing######
path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)      #학습데이터와 레이블


#########data split############
from random import shuffle,seed
from util import runCV1
seed(0)
numbers=list(range(len(d_mn)))
shuffle(numbers)
numbers[:5]

clf=DecisionTreeClassifier()
clf=clf.fit(d_mn,l_mn)
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
results=runCV1(clf,shuffled_data,shuffled_labels)
print(np.mean(results)) #0.85
#runCV로 샘플링하면 0.85의 성능이 나온다

