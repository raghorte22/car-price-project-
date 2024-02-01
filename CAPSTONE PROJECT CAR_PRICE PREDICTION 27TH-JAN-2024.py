#!/usr/bin/env python
# coding: utf-8

# # CAR PRICE PREDICTION

# In[ ]:


#car price prediction using ml


# ### importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings 
warnings.filterwarnings('ignore')


# In[3]:


dataset=pd.read_csv(r"E:\resume projects\car data.xls")


# In[4]:


dataset


# ### print columns name

# In[5]:


dataset.columns


# ### print top 5 rows

# In[6]:


dataset.head()


# ### display last 5 rows of the dataset

# In[7]:


dataset.tail()


# ### find shape of our dataset(number of rows and number of columns)

# In[8]:


dataset.shape


# In[9]:


print("Number of rows" ,dataset.shape[0])
print("Number of columns" , dataset.shape[1])


# ### get information about our dataset like the total number of rows ,total number of columns ,datatypes of each columns and memory requirement

# In[10]:


dataset.info()


# ### check null values in the dataset

# In[11]:


dataset.isnull()


# In[12]:


dataset.isnull().sum()


# ### get overall statistics about the dataset

# In[13]:


dataset.describe()


# ### data preprocessing

# In[14]:


dataset.head(1)


# In[15]:


import datetime
date_time=datetime.datetime.now()

dataset['Age']=date_time.year - dataset['Year']


# In[16]:


dataset.head()


# In[17]:


dataset.drop('Year',axis=1,inplace=True)


# In[18]:


dataset


# ### outlier removal

# In[19]:


sns.boxplot(dataset['Selling_Price'])


# ##### two datapoints are very far away from other datapoints so it will be consider as outlier

# In[20]:


sorted(dataset['Selling_Price'] ,reverse=True)


# In[21]:


(dataset['Selling_Price']>=33.0) & (dataset['Selling_Price']<=35.0)


# In[22]:


dataset[(dataset['Selling_Price']>=33.0) & (dataset['Selling_Price']<=35.0)]


# ##### this two are outlier

# In[23]:


dataset[~(dataset['Selling_Price'] >=33.0)  & (dataset['Selling_Price']<=35.0)]


# In[24]:


data=dataset[~(dataset['Selling_Price'] >=33.0)  & (dataset['Selling_Price']<=35.0)]


# In[25]:


data.shape


# ### Encoding the categorical columns

# In[26]:


dataset.head(1)


# In[27]:


dataset['Fuel_Type'].unique()


# In[28]:


dataset['Fuel_Type']=dataset['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})


# In[29]:


dataset['Fuel_Type'].unique()


# In[30]:


dataset['Seller_Type'].unique()


# In[31]:


dataset['Seller_Type']=dataset['Seller_Type'].map({'Dealer':0,'Individual':1})


# In[32]:


dataset['Seller_Type'].unique()


# In[33]:


dataset['Transmission'].unique()


# In[34]:


dataset['Transmission']=dataset['Transmission'].map({'Manual':0,'Automatic':1})


# In[35]:


dataset['Transmission'].unique()


# In[36]:


dataset.head()


# In[37]:


dataset.tail()


# In[38]:


dataset.info()


# ### store feature matrix in X and response (target) variable in Y

# In[39]:


x=dataset.drop(['Car_Name' , 'Selling_Price'], axis=1)
y=dataset['Selling_Price']


# In[40]:


# x is our independent variable


# In[41]:


x  


# In[42]:


# y is our target variable


# In[43]:


y


# ### splitting the datset into tarining set and testing set

# In[44]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# ### import the models

# In[45]:


dataset.head()


# In[46]:


pip install xgboost


# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# ### model training

# In[48]:


lr=LinearRegression()
lr.fit(x_train,y_train)

rf=RandomForestRegressor()
rf.fit(x_train,y_train)

xgb=GradientBoostingRegressor()
xgb.fit(x_train,y_train)

xg=XGBRegressor()
xg.fit(x_train,y_train)


# ### prediction of the test data 

# In[49]:


y_pred1=lr.predict(x_test)
y_pred2=rf.predict(x_test)
y_pred3=xgb.predict(x_test)
y_pred4=xg.predict(x_test)


# ### evaluating the algorithm

# In[50]:


from sklearn import metrics


# In[51]:


score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)


# In[52]:


print(score1,score2,score3,score4)


# In[53]:


final_data=pd.DataFrame({'Models':['LR' ,'RF' ,'GBR' ,'XG'],
                        'R2_SCORE':[score1,score2,score3,score4]})


# In[54]:


final_data


# In[55]:


sns.barplot(x=final_data['Models'], y=final_data['R2_SCORE'])


# ### save the model

# In[56]:


xg=XGBRegressor()
xg_final= xg.fit(x,y)


# In[57]:


import joblib 


# In[58]:


joblib.dump(xg_final,'car_price_predictor')


# In[59]:


model=joblib.load('car_price_predictor')


# In[60]:


model


# ### prediction on new data

# In[61]:


data_new=pd.DataFrame({
    'Present_Price' :5.59,
    'Kms_Driven' :27000,
    'Fuel_Type' :0,
    'Seller_Type' :0,
    'Transmission' :0,
    'Owner' :0,
    'Age' :8
} , index=[0])




# In[62]:


model.predict(data_new)


# ### GUI

# In[ ]:


from tkinter import *
import joblib
import pandas as pd

def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    
    model = joblib.load('car_price_predictor')
    data_new = pd.DataFrame({
        'Present_Price': p1,
        'Kms_Driven': p2,
        'Fuel_Type': p3,
        'Seller_Type': p4,
        'Transmission': p5,
        'Owner': p6,
        'Age': p7
    }, index=[0])
    
    result = model.predict(data_new)
    Label(master, text="Car Purchase amount").grid(row=8)
    Label(master, text=result).grid(row=10)
    print("Car Purchase amount", result[0])

master = Tk()
master.title("car price prediction using ml")
label = Label(master, text="car price prediction using ml", bg="black", fg="white").grid(row=0, columnspan=2)

Label(master, text="Present_Price").grid(row=1)
Label(master, text="Kms_Driven").grid(row=2)
Label(master, text="Fuel_Type").grid(row=3)
Label(master, text="Seller_Type").grid(row=4)
Label(master, text="Transmission").grid(row=5)
Label(master, text="Owner").grid(row=6)
Label(master, text="Age").grid(row=7)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
       
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




