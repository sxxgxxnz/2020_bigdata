from sklearn.svm import SVC


from random import shuffle,seed
from util import runCV1,load_mnist
import numpy as np

path='data/train_mnist.csv'
d_mn,l_mn=load_mnist(path)


numbers=list(range(len(d_mn)))

seed(0)
shuffle(numbers)
shuffled_data=d_mn[numbers]
shuffled_labels=l_mn[numbers]




params=[]

for k in ['linear','rbf','poly','sigmoid']:
    for C in [pow(10,x-3) for x in range(6)]:
        if k == 'linear':
            p=dict()
            p['kernel']=k
            p['C']=C
            params.append(p)
        elif k == 'rbf':
            for gc in [pow(10,x-3) for x in range(6)]:
                p=dict()
                p['kernel']=k
                p['C']=C
                p['gamma']=gc
                params.append(p)
        else:
            for gc in [pow(10,x-3) for x in range(6)]:
                for coef in [pow(10,x-3) for x in range(6)]:
                    p = dict()
                    p['kernel'] = k
                    p['C'] = C
                    p['gamma'] = gc
                    p['coef0']=coef
                    params.append(p)


means=dict()
for p in params:
    print(p)
    clf=SVC(**p)
    results=runCV1(clf,shuffled_data,shuffled_labels)
    means[str(p)]=np.mean(results)

ordered=sorted(means.items(),key=lambda x:x[1], reverse=True)
best_param=ordered[0]


from sklearn.model_selection import GridSearchCV

param1={'kernel':['linear'],
        'C':np.array([pow(10,x-3) for x in range(6)])}
param2={'kernel':['poly','sigmoid'],
        'C':np.array([pow(10,x-3) for x in range(6)]),
        'gamma':np.array([pow(10,x-3) for x in range(6)]),
        'coef0':np.array([pow(10,x-3) for x in range(6)])}
svc=SVC()
clf=GridSearchCV(svc,param1,cv=5,n_jobs=1)
clf.fit(shuffled_data,shuffled_labels)

prms=clf.cv_results_['params']
acc_means=clf.cv_results_['mean_test_score']    #각 파라미터의 결과

for mean,prm in zip(acc_means,prms):
    print("%0.3f for %r" %(mean,prm))


import time

svc = SVC()
clf = GridSearchCV(svc, param2, cv=5, n_jobs=8)

st=time.time()
clf.fit(shuffled_data, shuffled_labels)
et=time.time()
print(et-st)

prms = clf.cv_results_['params']
acc_means = clf.cv_results_['mean_test_score']  # 각 파라미터의 결과

for mean, prm in zip(acc_means, prms):
     print("%0.3f for %r" % (mean, prm))


###accuracysms gridsearch에서 자동으로 계산
###우리가 원하는 precision,recall,f1 계산

from sklearn.metrics import f1_score,precision_score,recall_score,make_scorer

f1_=make_scorer(f1_score,average='macro')
pr_=make_scorer(precision_score,average='macro')
re_=make_scorer(recall_score,average='macro')

clf=GridSearchCV(svc,param2,cv=5,n_jobs=4,
                 scoring={'f1':f1_,'pr':pr_,'re':re_},
                 refit=False)
st=time.time()
clf.fit(shuffled_data,shuffled_labels)
et=time.time()
print(et-st)