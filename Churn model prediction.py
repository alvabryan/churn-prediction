#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network

# ### Importing the libraries

# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[9]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Importing the dataset

# In[10]:


dataset = pd.read_csv('Churn_Modelling.csv')


# In[11]:


dataset.head()


# In[12]:


dataset.info()


# In[13]:


X = dataset.iloc[:,3:-1].values


# In[14]:


print(X)


# In[15]:


y = dataset.iloc[:,-1].values


# In[16]:


y


# ### Encoding categorical data

# Label Encoding the "Gender" column

# In[17]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[18]:


X[:,2] = le.fit_transform(X[:,2])


# One Hot Encoding the "Geography" column

# In[19]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))


# In[20]:


print(X)


# ### Splitting the dataset into the Training set and Test set

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)


# ###  Feature Scaling

# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Part 2 - Building the ANN

# ### Initializing the ANN

# In[23]:


ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))


# ## Part 3 - Training the ANN

# ### Compiling the ANN

# In[27]:


# Compililng the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
# Training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)


# ### Training the ANN on the Training set

# In[28]:





# ### Predicting one row

# In[32]:


prob = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print(prob > 0.5)


# ## Part 4 - Making the predictions and evaluating the model

# ### Predicting the Test set results

# In[30]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))


# ### Model Accuracy 

# In[34]:


# Importing the necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Calculating evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Printing the evaluation metrics
print("Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:




