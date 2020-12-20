#perceptron 0.775
from sklearn.linear_model import Perceptron
from util import runCV2,load_mnist
from random import shuffle
import time
import numpy as np


path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)

""" 0.775의 성능이 나온다.
clf=Perceptron(max_iter=500,n_jobs=3,eta0=0.1)
clf=clf.fit(norm_digits[:200],d_labels[:200])
pred=clf.predict(norm_digits[200:400])
correct=pred==d_labels[200:400]
acc=sum(correct)/len(correct)
"""

numbers=list(range(len(d_mn)))
shuffle(numbers)
clf=Perceptron(max_iter=500,n_jobs=3)
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
st=time.time()
results=runCV2(clf,shuffled_data,shuffled_labels,isAcc=False)
et=time.time()-st
print(et)
print(np.mean(results)) #40.15