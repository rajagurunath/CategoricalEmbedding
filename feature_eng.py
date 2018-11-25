
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as fs


# In[3]:


df=pd.read_csv(r'train.csv')
df.shape


# In[7]:


df.dtypes,df.describe()


# In[8]:


df.isnull().sum()


# In[10]:


df['class'].value_counts().plot(kind='bar')


# In[11]:


df.corr()


# In[13]:


df.corr()['class'].plot(kind='bar')


# In[14]:


df.nunique()


# In[19]:


age_bins=pd.cut(df.age,bins=10)


# In[21]:


df['age_bins']=age_bins.cat.codes


# In[22]:


df.corr()['class'].plot(kind='bar')


# In[24]:


cont_df=df.select_dtypes('float64')


# In[25]:


cont_df.plot()


# In[58]:


for col in cont_df.columns:
    fig=plt.figure(figsize=(20,10))
    plt.scatter(cont_df[col].index,cont_df[col],c=df['class'])
    plt.title(col)
    plt.show()


# In[61]:


plt.hist(one['chest'])


# In[65]:


one,zero=cont_df[df['class']==1].copy(),cont_df[df['class']==0].copy()
for col in cont_df.columns:
    fig=plt.figure(figsize=(20,10))
    plt.hist(one[col],alpha=0.5,color='red')
    plt.hist(zero[col],alpha=0.5,color='green')
    plt.title(col)
    plt.show()


# In[4]:


cat_df=df.select_dtypes('int64')


# In[36]:


cat_df.groupby('class').agg('count')


# In[37]:


cat_df.columns


# In[43]:


cat_df.groupby(['class','sex','fasting_blood_sugar','resting_electrocardiographic_results','exercise_induced_angina','slope','number_of_major_vessels','thal']).agg('count')


# In[52]:


cat_df.groupby(['class','sex','fasting_blood_sugar','resting_electrocardiographic_results','exercise_induced_angina','slope','number_of_major_vessels','thal']).agg('count').plot(kind='bar')
plt.xticks(rotation='vertical')
plt.show()


# In[49]:


cat_df.groupby(['class','sex','fasting_blood_sugar','resting_electrocardiographic_results']).agg('count').plot(kind='bar')
plt.xticks(rotation='vertical')
plt.show()


# In[8]:


cat_df.columns[1:-1]


# In[10]:


ss=cat_df.groupby(['class','slope'])['ID'].count()


# In[9]:


for col in cat_df.columns[1:-1]:
    ss=cat_df.groupby(['class',col])['ID'].count()
    fig, ax = plt.subplots()    
    ax.barh(list(range(ss.shape[0])),ss.values)
    for i, v in enumerate(ss):
        #ax.barh(ss.index,ss.values)

        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    plt.ylabel(ss.index.values)
    plt.title(col)
    plt.show()


# In[29]:


import seaborn as sns


# In[9]:


sns.countplot(cat_df['slope'])


# # Feature Engineering with Model building 

# In[13]:


import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split


# In[12]:


clf=xgb.XGBClassifier()


# In[19]:


df.iloc[:,1:-1].columns,df.iloc[:,-1].name


# In[21]:


X_train, X_test, y_train, y_test=train_test_split(df.iloc[:,1:-1],df.iloc[:,-1])


# In[22]:


clf.fit(X_train,y_train)


# In[23]:


pred=clf.predict(X_test)


# In[30]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
sns.heatmap(cm)


# In[31]:


plot_importance(clf)


# In[32]:


plot_tree(clf)

