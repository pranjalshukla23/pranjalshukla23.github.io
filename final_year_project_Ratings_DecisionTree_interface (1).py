#!/usr/bin/env python
# coding: utf-8

# # 1. Import libraries

# In[30]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# # 2. Acquire the data

# In[31]:


df=pd.read_csv("ratings.csv")
df.head()


# # 3. Preprocess the data 

# In[32]:


df.info()


# In[33]:


df=df[["Team","Player","Tournament","Matches","Ratings","team_id"]]


# In[34]:


df.head()


# In[35]:


df.info()


# In[36]:


df.isna().sum()


# In[37]:


df.describe()


# In[38]:


sns.countplot(x="Ratings",data=df)


# In[39]:


X=df.drop(["Team","Player","Tournament","Ratings"],axis=1)
y=df["Ratings"]


# In[40]:


X


# In[41]:


y


# In[42]:


X.shape


# In[43]:


y.shape


# # 5. Split the data into training and testing 

# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)


# In[45]:


X_train.shape


# In[46]:


X_test.shape


# In[47]:


y_test.shape


# In[48]:


y_train.shape


# In[49]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# # 6. Train the model

# In[50]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=112)
result=model.fit(X_train,y_train)


# In[51]:


y_pred=result.predict(X_test)


# In[52]:


y_pred


# # 7. Deploy the model

# In[53]:


predictions=result.predict([[10,4]])
predictions
if(predictions[0]==4):
    print("Great Form")
elif (predictions[0]==5):
      print("Excellent form")
elif (predictions[0]==3):
    print("Good form")
elif (predictions[0]==2):
    print("In-form")
elif (predictions[0]==1):
    print("Out of form")


# In[54]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[55]:


import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 200,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Get the Ratings:')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Enter the number of matches played:')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry (root) 
canvas1.create_window(200, 150, window=entry1)


canvas2 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas2.pack()


label4 = tk.Label(root, text='Enter the Team ID:')
label4.config(font=('helvetica', 10))
canvas2.create_window(200, 10, window=label4)

entry2 = tk.Entry (root) 
canvas2.create_window(200, 50, window=entry2)




def Score ():
    
    x1 = entry1.get()
    x2=entry2.get()
    
    # Testing with a custom input
    import numpy as np
    new_prediction = result.predict(sc.transform(np.array([[x1,x2]])))

    word=str(new_prediction[0])
    words="Predicted Rating of player is:"+word
   
    
    label1 = tk.Label(root, text=words,font=('helvetica', 13, 'bold'))
    canvas2.create_window(200, 113, window=label1)
    
button1 = tk.Button(text='Get the Predicted Rating', command=Score, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas2.create_window(200, 90, window=button1)

root.mainloop()
    
    
    
    
    


# In[56]:


def predictRating(number1,number2):
    import numpy as np
    answer = result.predict(sc.transform(np.array([[number1,number2]])))
    return answer[0]

