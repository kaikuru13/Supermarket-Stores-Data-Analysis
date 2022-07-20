#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install plotly')
get_ipython().system('pip install seaborn')


# In[ ]:


import pandas as pd 
import numpy as np 
import plotly.express as px 
import seaborn as sns
import missingno as msno
import os 


# In[6]:


df = pd.read_csv("C:\\Users\\kaila\\Desktop\\Stores.csv")


# In[7]:


df


# In[8]:


df.info()


# In[9]:


df.head()


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated()


# In[17]:


df.describe()


# In[22]:


df['Total_Monthly_Customers_Count'] = df['Daily_Customer_Count'] * 30
df['Avg_Customer_Spend_Day'] = df['Store_Sales'] / df['Total_Monthly_Customers_Count']
df['Avg_Customer_Spend_Month'] = df['Store_Sales'] / df['Daily_Customer_Count']
df['Avg_Daily_Sales'] = df['Store_Sales'] / 30


# In[24]:


df.head()


# In[25]:


import matplotlib.pyplot as plt
df.Daily_Customer_Count.hist(bins = 50, figsize=(12,8))
plt.show()


# In[26]:


df.hist(figsize=(12,8));
plt.grid(False)


# In[27]:


correlation = df.corr()
print(correlation['Store_Sales'].sort_values(ascending = False),'\n')


# In[28]:


k= 10
cols = correlation.nlargest(k,'Store_Sales')['Store_Sales'].index
print(cols)
cm = np.corrcoef(df[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap="YlGnBu" ,
 linecolor="b",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
plt.show()


# In[29]:


Yfac = df["Store_Sales"]
Xfac = df["Daily_Customer_Count"]
plt.figure(figsize=(12,8))
plt.scatter(Xfac,Yfac,s=15)
plt.xlabel('Count of Customers')
plt.ylabel('Sales in $')
plt.title('Customers & Sales of All Stores')
plt.xlim(200, 2000)
plt.yscale("linear")
plt.grid(True)


# In[30]:


x = df[['Items_Available']]
y = df[['Store_Sales']]
_ = plt.figure(figsize=(12,8))  
_ = plt.scatter(x , y)
_ = plt.xlabel('Items Available')
_ = plt.ylabel('Sales')
_ = plt.xticks(np.arange(0, 2800 , step = 200)) 
_ = plt.yticks(np.arange(0 ,120000 , step = 20000), ['20K' , '40K' , '60K' , '80K' , '100K' , '120K']) 

plt.plot()


# In[31]:


df.sort_values('Store_Sales', ascending=False)

