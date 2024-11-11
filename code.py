#!/usr/bin/env python
# coding: utf-8

# In[225]:


import numpy as np
t=[123,58,9,3,9,3,5]
sum1=np.sum(t)
print("sum value is:",sum1)
max=np.max(t)
print("maximum value is:",max)
min=np.min(t)
print("minimum value is:",min)
med=np.median(t)
print("median value is:",med)
mean=np.mean(t)
print("mean value is:",mean)
pro=np.prod(t)
print("produce value is:",pro)
std=np.std(t)
print("standard deviation is:",std)
argmin=np.argmin(t)
print("argument min value is:",argmin)
argmax=np.argmax(t)
print("argumnet max:",argmax)
coc=np.corrcoef(t)
print("correlation coefient:",coc)


# In[226]:


lambda_cube=lambda y:y*y*y
lambda_cube(5)


# In[227]:


from functools import reduce
def sum(x,y):
    return x+y    
l=[45,8,9,6,9]
l1=(reduce(sum,l))
l1


# In[228]:


def oddeven(x):
    if(x%2==0):
        return True
    else:
        return False
l=[45,8,9,6,9]
l1=(list(filter(oddeven,l)))
l1


# In[229]:


def add(x):
    return x+4
l=[45,8,9,6,9]
l1=(list(map(add,l)))
l1


# In[230]:


import pandas as pd
data=pd.DataFrame({'value':[12,4,5,8,9,3,89]})
mean=np.mean(data['value'])
print("mean value is:",mean)
std=np.std(data['value'])
print("std value is:",std)
threshold=1
outlier=[]
for i in data['value']:
    z=(i-mean)/std
    if z > threshold:
        outlier.append(i)
print("outlier using z score method:",outlier)


# In[231]:


q1=data['value'].quantile(0.25)
q3=data['value'].quantile(0.75)
iqr=q3-q1
lowerbound=q1-1.5*iqr
upperbound=q3+1.5*iqr
outlier=data[(data['value'] < lowerbound) | (data['value'] > upperbound)]
print("outlier using iqr method:",outlier)


# In[232]:


mean=data['value'].mean()
print("mean value is:",mean)
for i in data['value']:
    if i > lowerbound or i < upperbound:
        data['value']=(data['value'].replace(i,mean))


# In[233]:


med=data['value'].median()
print("median value is:",med)
for i in data['value']:
    if i > lowerbound or i < upperbound:
        data['value']=(data['value'].replace(i,med))


# In[234]:


for i in data['value']:
    if i > lowerbound or i < upperbound:
        data['value']=(data['value'].replace(i,20))
data['value']


# In[235]:


import pandas as pd
df=pd.read_csv('customer.csv')
df


# In[236]:


import matplotlib.pyplot as plt
df.plot("age","salary",kind="scatter")


# In[237]:


df['age'].plot(kind="bar")


# In[238]:


df['age'].plot(kind="box")


# In[239]:


df['age'].plot(kind="hist")


# In[240]:


import seaborn as sns
sns.pairplot(data=df)
plt.show()


# In[241]:


df=pd.read_csv('iris.csv')
df


# In[242]:


plt.scatter(x="sepallength",y="sepalwidth",data=df)
plt.title("scatter plot")
plt.xlabel("seppel length")
plt.ylabel("sepal width")
plt.show()


# In[243]:


plt.bar(df["sepallength"],df["sepalwidth"],data=df)
plt.title("bar plot")
plt.xlabel("seppel length")
plt.ylabel("sepal width")
plt.show()


# In[244]:


sns.boxplot(df["sepallength"])
plt.title("box plot")
plt.xlabel("seppel length")
plt.ylabel("sepal width")
plt.show()


# In[245]:


plt.hist(df["sepallength"])
plt.title("hist plot")
plt.xlabel("seppel length")
plt.ylabel("sepal width")
plt.show()


# In[246]:


import pandas as pd
data={'ename':['kishan','kushi','rishab','krithika'],
      'salary':[2500,89000,45700,58300],
      'department':['cs','ec','is','aiml'],
      'age':[28,21,32,24]}
df=pd.DataFrame(data)
df


# In[251]:


df.loc[df['ename']=="kishan",'salary']+1000


# In[249]:


df


# In[201]:


df.describe()


# In[202]:


df.isnull()


# In[203]:


df['t']=[45,8,96,96]


# In[204]:


df


# In[205]:


df.drop(columns='t')


# In[206]:


df.loc[5]=['anu',25588,'cs',32,58]


# In[207]:


df


# In[208]:


df.drop(index=5,axis=0)


# In[209]:


df.head()


# In[210]:


print(df)


# In[211]:


import missingno as msn
msn.bar(df)
plt.show()


# In[212]:


df['salary'].value_counts()


# In[213]:


df.groupby('salary')['ename'].sum()


# In[214]:


avg=df.pivot_table(values='salary',index='department',columns='ename',aggfunc='mean')
avg


# In[215]:


sum=df.pivot_table(values='salary',index='department',columns='ename',aggfunc='sum')
sum


# In[216]:


from sklearn.datasets import load_iris
df=load_iris()


# In[217]:


df


# In[218]:


x=df.data
y=df.target


# In[219]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[220]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train,y_train)


# In[252]:


y_pred=lr.predict(x_test)


# In[259]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=r2_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[262]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train,y_train)


# In[263]:


y_pred=lr.predict(x_test)


# In[264]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=accuracy_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[267]:


cf=classification_report(y_test,y_pred)
print(cf)


# In[268]:


from sklearn.ensemble import RandomForestClassifier
lr= RandomForestClassifier()
lr.fit(x_train,y_train)


# In[269]:


y_pred=lr.predict(x_test)


# In[270]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=accuracy_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[271]:


from sklearn.ensemble import RandomForestRegressor
lr=RandomForestRegressor()
lr.fit(x_train,y_train)


# In[272]:


y_pred=lr.predict(x_test)


# In[274]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=r2_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[278]:


from sklearn.tree import DecisionTreeClassifier
lr= DecisionTreeClassifier()
lr.fit(x_train,y_train)


# In[279]:


y_pred=lr.predict(x_test)


# In[280]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=r2_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[281]:


from sklearn.tree import DecisionTreeRegressor
lr= DecisionTreeRegressor()
lr.fit(x_train,y_train)


# In[282]:


y_pred=lr.predict(x_test)


# In[283]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=r2_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[285]:


from sklearn.svm import SVC
lr= SVC(kernel='linear',gamma=0.5)
lr.fit(x_train,y_train)


# In[286]:


y_pred=lr.predict(x_test)


# In[287]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=accuracy_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[295]:


from sklearn.neighbors import KNeighborsClassifier
lr= KNeighborsClassifier(n_neighbors=3)
lr.fit(x_train,y_train)


# In[296]:


y_pred=lr.predict(x_test)


# In[297]:


from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
acc=accuracy_score(y_test,y_pred)
print("accuracy score is:",acc)


# In[301]:


df=pd.read_csv('data.csv')
df


# In[302]:


df.isnull().sum()


# In[304]:


df.interpolate()


# In[311]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=30,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("no of cluster")
plt.ylabel("count")
plt.show()


# In[312]:


km1=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=30,random_state=0)
y_means=km1.fit_predict(x)


# In[313]:


y_means


# In[318]:


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='pink',label='c1:kanjoos')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='magenta',label='c2:backra')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='cyan',label='c3:pokiri')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='yellow',label='c4:average')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='orange',label='c5:intelligent')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],s=50,c="pink",label="centroid")
plt.title("kmeans clustering")
plt.xlabel("no of cluster")
plt.ylabel("count")
plt.legend()
plt.show()


# In[319]:


km1.cluster_centers_


# In[ ]:




