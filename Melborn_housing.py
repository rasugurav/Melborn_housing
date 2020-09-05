#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sns


# In[3]:


df=pd.read_csv("Melbourne_housing.csv")
df.head()


# In[4]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.describe(include="all")


# In[9]:


# Data Wrangling
missing_data=df.isnull()
missing_data.head()


# In[12]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())


# In[13]:


avg_price=df["Price"].astype("float").mean(axis=0)


# In[15]:


print("Avg of price:",avg_price)


# In[16]:


df["Price"].replace(np.nan,avg_price,inplace=True)


# In[17]:


df.Price


# In[19]:


avg_distance=df["Distance"].astype("float").mean(axis=0)
avg_distance


# In[20]:


df["Distance"].replace(np.nan,avg_distance,inplace=True)
df.Distance


# In[21]:


df["Postcode"].value_counts().idxmax()


# In[22]:


df["Postcode"].replace(np.nan,'3073',inplace=True)


# In[23]:


avg_bedroom=df["Bedroom2"].astype("float").mean(axis=0)
avg_bedroom


# In[24]:


df["Bedroom2"].replace(np.nan,'3',inplace=True)
df["Bedroom2"]


# In[26]:


avg_bathroom=df["Bathroom"].astype('float').mean(axis=0)
avg_bathroom


# In[27]:


df["Bathroom"].replace(np.nan,'2',inplace=True)


# In[28]:


df["Bathroom"]


# In[30]:


avg_landsize=df["Landsize"].astype("float").mean(axis=0)
avg_landsize


# In[31]:


df["Landsize"].replace(np.nan,avg_landsize,inplace=True)
df["Landsize"]


# In[32]:


avg_buildarea=df["BuildingArea"].astype("float").mean(axis=0)


# In[33]:


avg_buildarea


# In[34]:


df["BuildingArea"].replace(np.nan,avg_buildarea,inplace=True)


# In[35]:


df["YearBuilt"].value_counts().idxmax()


# In[36]:


df["YearBuilt"].replace(np.nan,'1970',inplace=True)


# In[40]:


avg_car=df["Car"].astype("float").mean(axis=0)
df["Car"].replace(np.nan,'2',inplace=True)


# In[41]:


df.dropna(subset=["CouncilArea"],axis=0,inplace=True)


# In[43]:


df.dropna(subset=["Longtitude"],axis=0,inplace=True)


# In[44]:


df.dropna(subset=["Lattitude"],axis=0,inplace=True)


# In[48]:


df.dropna(subset=["Regionname"],axis=0,inplace=True)


# In[50]:


avg_property_count=df["Propertycount"].astype("float").mean(axis=0)


# In[51]:


avg_property_count


# In[52]:


df["Propertycount"].replace(np.nan,avg_property_count,inplace=True)


# In[53]:


df.head()


# In[57]:


plt.hist(df["Price"])
plt.xlabel("Price")


# In[58]:


bins=np.linspace(min(df["Price"]),max(df["Price"]),4)
bins


# In[62]:


df.corr()


# In[65]:


df[['Distance','Landsize','BuildingArea']].corr()


# In[67]:


df[['Price','BuildingArea']].corr()


# In[68]:


import seaborn as sns


# In[69]:


sns.regplot(x='Rooms',y="Price",data=df)


# In[70]:


sns.regplot(x="Distance",y="Price",data=df)


# In[71]:


sns.regplot(x="Landsize",y="Price",data=df)


# In[73]:


sns.regplot(x="BuildingArea",y="Price",data=df)


# In[74]:


sns.boxplot(x="Rooms",y="Price",data=df)


# In[75]:


#Linear Regression


# In[77]:


from sklearn.linear_model import LinearRegression


# In[78]:


lm=LinearRegression()
lm


# In[79]:


X=df[["Bedroom2"]]
y=df["Price"]
lm.fit(X,y)


# In[81]:


yhat=lm.predict(X)
yhat[0:5]


# In[82]:


lm.intercept_
lm.coef_


# In[83]:


z=df[["Distance","BuildingArea","Distance","Landsize"]]
lm.fit(z,df["Price"])


# In[84]:


lm.intercept_
lm.coef_


# In[86]:


lm.score(z,df["Price"])


# In[87]:


y_predict_multifit=lm.predict(z)


# In[91]:


from sklearn.metrics import mean_squared_error

mean_squared_error(df["Price"],y_predict_multifit)


# In[97]:


new_input=np.arange(1,100,1).reshape(-1,1)


# In[98]:


lm.fit(X,y)


# In[99]:


lm


# In[100]:


yhat=lm.predict(new_input)
yhat[0:5]


# In[101]:


plt.plot(new_input,yhat)
plt.show()


# In[102]:


##Training and Testing


# In[103]:


y_data=df["Price"]
x_data=df.drop("Price",axis=1)


# In[104]:


from sklearn.model_selection import train_test_split


# In[105]:


x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.15,random_state=1)


# In[106]:


x_test.shape[0]
x_train.shape[0]


# In[108]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[113]:


lr.fit(x_train[["Bathroom"]],y_train)


# In[114]:


lr.score(x_test[["Bathroom"]],y_test)


# In[115]:


lr.score(x_train[["Bathroom"]],y_train)


# In[116]:


lr.fit(x_train[["Distance"]],y_train)


# In[117]:


lr.score(x_test[["Distance"]],y_test)


# In[119]:


lr.score(x_train[["Distance"]],y_train)


# In[120]:


## CRoss Validation


# In[121]:


from sklearn.model_selection import cross_val_score


# In[122]:


cross=cross_val_score(lr,x_data[["Distance"]],y_data,cv=4)


# In[123]:


cross


# In[124]:


cross.mean()


# In[126]:


cross.std()


# In[127]:


from sklearn.model_selection import cross_val_predict


# In[128]:


yhat=cross_val_predict(lr,x_data[["Distance"]],y_data,cv=4)
yhat[0:5]


# In[129]:


##Over fitting and Underfitting


# In[130]:


lr.fit(x_train[["Distance","Rooms","Bathroom"]],y_train)


# In[131]:


y_hat=lr.predict(x_train[["Distance","Rooms","Bathroom"]])


# In[132]:


y_hat[0:5]


# In[133]:


y_hat_test=lr.predict(x_test[["Distance","Rooms","Bathroom"]])
y_hat_test[0:5]

