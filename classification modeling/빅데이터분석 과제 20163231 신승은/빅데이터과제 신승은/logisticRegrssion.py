from util import load_mnist
import numpy as np
path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)

from sklearn.linear_model import LogisticRegression as lr
from random import shuffle
from util import runCV2
numbers=list(range(len(d_mn)))
shuffle(numbers)
clf=lr()


shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]
results=runCV2(clf,shuffled_data,shuffled_labels,isAcc=False)
print(np.mean(results))