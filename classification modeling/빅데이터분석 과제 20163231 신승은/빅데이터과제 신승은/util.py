import numpy as np

def get10fold(data,turn):
    tot_length=len(data)
    each=int(tot_length/10)
    mask=np.array([True if each*turn<= i <each*(turn+1)else False for i in list(range(tot_length))])
    return data[~mask], data[mask]

def get_one_fold(data,turn,fold=10):
    tot_length=len(data)
    each=int(tot_length/fold)
    mask=np.array([True if each*turn<=i<each*(turn+1) else False for i in list(range(tot_length))])
    return data[~mask],data[mask]


def runCV2 (clf,shuffled_data, shuffled_labels,fold=10, isAcc=True):
    from sklearn.metrics import precision_recall_fscore_support
    results=[]

    for i in range(fold):
        train_data, test_data = get_one_fold(shuffled_data, i,fold=fold)
        train_labels, test_labels = get_one_fold(shuffled_labels, i,fold=fold)

        clf= clf.fit(train_data , train_labels)
        pred= clf.predict(test_data)
        correct =pred ==test_labels
        if isAcc:
            acc=sum([1 if x == True else 0 for x in correct])/len(correct)
            results.append(acc)
        else:
            results.append(precision_recall_fscore_support(pred,test_labels))

    return results

def runCV1(clf,data,labels,fold=10):
    accuracies=[]
    for i in range(fold):
        train_data,test_data=get10fold(data,i)
        train_labels,test_labels=get10fold(labels,i)
        clf=clf.fit(train_data,train_labels)
        pred=clf.predict(test_data)
        correct=pred==test_labels
        acc=sum([1 if x==True else 0 for x in correct])/len(correct)
        accuracies.append(acc)
    return accuracies

def load_mnist(path):
    f=open(path,'r')
    f.readline()
    digits=[]
    d_labels=[]
    for line in f.readlines():
        splitted=line.replace("\n","").split(",")
        digit=np.array(splitted[1:],dtype=np.float32)
        label=int(splitted[0])
        digits.append(digit)
        d_labels.append(label)
    digits=np.array(digits)
    d_labels=np.array(d_labels)
    norm_digits=digits/255

    return norm_digits,d_labels



