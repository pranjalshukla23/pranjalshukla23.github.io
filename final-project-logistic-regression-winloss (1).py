#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
# Importing the dataset
dataset = pd.read_csv('odi-winloss.csv')
X = dataset.iloc[:,[7,8,9,12,13,14]]
y = dataset.iloc[:, 15]


# In[66]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(27, 6))
sns.barplot(x="bat_team",y="win",data=dataset)


# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(27, 6))
sns.barplot(x="bowl_team",y="win",data=dataset)


# In[68]:


X.head()


# In[69]:


y.head()


# In[70]:


import pandas as pd
# Importing the dataset
dataset = pd.read_csv('odi-winloss.csv')
X = dataset.iloc[:,[7,8,9,12,13,14]].values
y = dataset.iloc[:, 15].values


# In[71]:


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[72]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[73]:


# Training the dataset
from sklearn.linear_model import LogisticRegression
lin = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
lin.fit(X_train,y_train)


# In[74]:


# Testing the dataset on trained model
from sklearn.metrics import accuracy_score
y_pred = lin.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
print(accuracy*100)


# In[75]:


# Testing with a custom input
import numpy as np
new_prediction = lin.predict_proba(sc.transform(np.array([[100,2,10,23,52,200]])))
print(new_prediction[0][1]*100)


# In[ ]:


import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 200,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Calculate the Predicted chances of winning:')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Enter the runs:')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry (root) 
canvas1.create_window(200, 150, window=entry1)


canvas2 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas2.pack()


label4 = tk.Label(root, text='Enter the Wickets:')
label4.config(font=('helvetica', 10))
canvas2.create_window(200, 10, window=label4)

entry2 = tk.Entry (root) 
canvas2.create_window(200, 50, window=entry2)

canvas3 = tk.Canvas(root, width = 400, height = 70,  relief = 'raised')
canvas3.pack()


label5 = tk.Label(root, text='Enter the Overs:')
label5.config(font=('helvetica', 10))
canvas3.create_window(200, 10, window=label5)

entry3 = tk.Entry (root) 
canvas3.create_window(200, 30, window=entry3)

canvas4 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas4.pack()


label6 = tk.Label(root, text='Enter the Striker runs:')
label6.config(font=('helvetica', 10))
canvas4.create_window(200, 10, window=label6)

entry4 = tk.Entry (root) 
canvas4.create_window(200, 30, window=entry4)

canvas5 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas5.pack()


label7 = tk.Label(root, text='Enter the Non-striker runs:')
label7.config(font=('helvetica', 10))
canvas5.create_window(200, 10, window=label7)

entry5 = tk.Entry (root) 
canvas5.create_window(200, 30, window=entry5)

canvas6 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas6.pack()


label8= tk.Label(root, text='Enter the target :')
label8.config(font=('helvetica', 10))
canvas6.create_window(200, 10, window=label8)

entry6 = tk.Entry (root) 
canvas6.create_window(200, 30, window=entry6)





def Score ():
    
    x1 = entry1.get()
    x2=entry2.get()
    x3=entry3.get()
    x4=entry4.get()
    x5=entry5.get()
    x6=entry6.get()
    
    # Testing with a custom input
    import numpy as np
    new_prediction = lin.predict_proba(sc.transform(np.array([[x1,x2,x3,x4,x5,x6]])))
    result1=new_prediction[0][1]*100
   
    result="The chances of winning is:"+str(result1)
    
    label1 = tk.Label(root, text=result,font=('helvetica', 13, 'bold'))
    canvas6.create_window(200, 100, window=label1)
    
button1 = tk.Button(text='Get the Predicted chances of winning', command=Score, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas6.create_window(200, 70, window=button1)

root.mainloop()
    
    
    
    
    

