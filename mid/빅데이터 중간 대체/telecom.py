
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

plt.show()
mpl.rc('font',family='Malgun Gothic')



df=pd.read_csv('data/telco.csv')
df.head()


############### 데이터 전처리 ###############

np_mincallcost=df['분당통화요금'].values

mask = np_mincallcost != '?'
mask_na = [not(x) for x in mask]

#분당통화요금의 결측치를 제거하고 다시 불러옴
df=pd.read_csv('data/telco.csv',na_values='?').dropna()
df.shape




#컬럼별 정보표시
df.info()


#사용하지 않는 컬럼제거
df.drop(['id','고객ID','개시일'],axis=1,inplace=True)



#연속형 변수에 대한 기초통계량 산출
df.describe()


############## 시각화분석 ##############



#상관관계 그래프
corr = df.corr()
plt.figure(figsize=(12,10))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)


#종속변수의 항목별 수 확인
sns.countplot(df['이탈여부'])
df['이탈여부'].value_counts()



#이탈여부와 단선횟수의 박스그래프
plt.figure(figsize=[8,6])
sns.boxplot(x='이탈여부', y='단선횟수', data=df)

#핸드셋종류와 이탈여부별 분당 통화요금의 평균분포
plt.figure(figsize=[10,5])
sns.barplot(x='핸드셋',y='분당통화요금',data=df,hue='이탈여부')


#통화품질 불만여부와 이탈여부별 평균납부요금의 선그래프
plt.figure(figsize=[10,5])
sns.lineplot(x='이탈여부',y='평균납부요금',data=df,hue='통화품질불만')


############ 모델링 ##############

#범주형 변수들 매핑
cate=['지불방법', '요금제', '이탈여부','핸드셋', '통화량구분','납부여부','미사용','통화품질불만','성별']


#범주형 변수들을 숫자형으로 변환하기 위해 클래스 분포를 확인
for i in cate:
    print('---------------',i,'--------------')
    print(df[i].value_counts())

#미사용과 지불방법은 한가지 항목밖에 없으므로 제거
df.drop(['미사용','지불방법'],axis=1,inplace=True)

#범주형 변수들을 항목별로 숫자형 변수들로 바꿔줌
df['요금제']=df['요금제'].replace('CAT 200',1).replace('CAT 100',2).replace('Play 100',3).replace('Play 300',4).replace('CAT 50',5)
df['이탈여부']=(df['이탈여부'].values == '유지').astype(np.int)
df['핸드셋']=df['핸드셋'].replace('S50',1).replace('BS110',2).replace('ASAD90',3).replace('S80',4).replace('CAS30',5).replace('WC95',6).replace('ASAD170',7).replace('BS210',8).replace('SOP20',9).replace('SOP10',10).replace('CAS60',11)
df['통화량구분']=df['통화량구분'].replace('중',1).replace('중고',2).replace('중저',3).replace('고',4).replace('저',5)
df['납부여부']=df['납부여부'].replace('OK',1).replace('High CAT 100',2).replace('High CAT 50',3).replace('High Play 100',4)
df['통화품질불만']=(df['통화품질불만'].values == 'T').astype(np.int)
df['성별']=(df['성별'].values=='남').astype(np.int)


df.info()


############### 모델 구축 ###############

from sklearn.ensemble import RandomForestClassifier
from random import shuffle, seed
from sklearn.metrics import accuracy_score


from random import shuffle, seed
seed(3)
numbers = list(range(len(df)))
shuffle(numbers)

#X는 독립변수, Y는 종속변수
X=df.drop('이탈여부',axis=1)
Y=df['이탈여부']

split_point = int(len(df)/10*7)  #7:3으로 데이터를 분할(학습=7, 테스트=3)
X_train = X.iloc[numbers[:split_point],]# 학습 데이터
Y_train = Y.iloc[numbers[:split_point]] # 학습 레이블
X_test = X.iloc[numbers[split_point:]] # 평가 데이터
Y_test = Y.iloc[numbers[split_point:]] # 평가 레이블


#RandomForest 모델 생성 후 학습 셋으로 학습 시킨뒤 test셋을 이용하여 예측한다.
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
pred=rf.predict(X_test) # 평가 데이터로 예측
correct = pred==Y_test # 예측 결과 비교
acc = sum([1 if x == True else 0 for x in correct])/len(correct)
acc

#10-fold CV
def get10fold(data, turn):
    import numpy as np
    tot_length= len(data)
    each = int(tot_length/10)
    mask = np.array ([True if each*turn<=i< each*(turn+1) else False
                      for i in list(range( tot_length ))])
    return data[~mask], data[mask]

#학습데이터와 테스트 데이터를 10fold-cv수행 -> 평균 정확도
seed(3)
numbers= list(range(len(df)))
shuffle(numbers)
shuffled_data=X.iloc[numbers]
shuffled_labels=Y.iloc[numbers]
rf= RandomForestClassifier()
accuracies =[]
for i in range(10):
    train_data, test_data = get10fold(shuffled_data , i)
    train_labels, test_labels =get10fold(shuffled_labels , i)
    rf= rf.fit(train_data , train_labels)
    pred=rf.predict(test_data)
    correct =pred == test_labels
    acc = sum([1 if x == True else 0 for x in correct])/len(correct)
    accuracies.append(acc)
    
print(np.mean (accuracies))



################ 모델 튜닝 ################

#하이퍼 파라미터 튜닝

#GridSearchCV를 통한 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV

#RandomForest의 하이퍼 파라미터
RandomForestClassifier()


model=RandomForestClassifier()
params={'n_estimators':[100,200,300],'max_depth':[None,3,4,7],'class_weight':[None,'balanced'],'min_samples_leaf':[1,2,3]}

rf=GridSearchCV(model,param_grid=params, cv=5)

rf.fit(X_train,Y_train)

#각 파라미터 조합에 대한 정확도
prms= rf.cv_results_['params']
acc_means= rf.cv_results_['mean_test_score']
acc_stds= rf.cv_results_['std_test_score']
for mean , std , prm in zip(acc_means , acc_stds , prms):
    print("%0.3f (+/-%0.03f) for %r"%(mean , std * 2, prm))

#가장 높은 정확도를 가진 파라미터
rf.best_params_

#최적의 하이퍼 파라미터일때의 정확도
rf.best_score_


#교정된 파라미터를 적용한 모델로 10fold-cv수행 -> 평균 정확도
seed(2)
numbers= list(range(len(df)))
shuffle(numbers)
shuffled_data=X.iloc[numbers]
shuffled_labels=Y.iloc[numbers]
rf= RandomForestClassifier(class_weight='balanced',min_samples_leaf=3,n_estimators=200)
accuracies =[]
for i in range(10):
    train_data, test_data = get10fold(shuffled_data , i)
    train_labels, test_labels =get10fold(shuffled_labels , i)
    rf= rf.fit(train_data , train_labels)
    pred=rf.predict(test_data)
    correct =pred == test_labels
    acc = sum([1 if x == True else 0 for x in correct])/len(correct)
    accuracies.append(acc)
    
print(np.mean(accuracies))

