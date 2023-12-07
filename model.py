#!/usr/bin/env python
# coding: utf-8

# # Description:
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
#
# Attributes:
# 1. Glucose Level
# 2. BMI
# 3. Blood pressure
# 4. Pregnancies
# 5. Skin thickness
# 6. Insulin
# 7. Diabetes pedigree function
# 8. Age
# 9. Outcome

# # Step 0: Import libraries and Dataset

# In[1]:


import pandas as pd


import warnings
warnings.filterwarnings('ignore')

import pickle

# In[2]:


dataset = pd.read_csv('diabetes.csv')



# # Step 3: Data Preprocessing

# In[13]:


dataset_X = dataset.iloc[:,[1,2,3, 4, 5,6, 7]].values
dataset_Y = dataset.iloc[:,8].values


# In[14]:


dataset_X


# In[15]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


# In[16]:


dataset_scaled = pd.DataFrame(dataset_scaled)


# In[17]:


X = dataset_scaled
Y = dataset_Y


# In[18]:


X


# In[19]:


Y


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset['Outcome'] )


# # Step 4: Data Modelling

# In[25]:


# Import the KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create an instance of KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=24)  # You can choose the number of neighbors

# Fit the model to the training data
knn.fit(X_train, Y_train)

# Evaluate the model on the test set
accuracy_knn = knn.score(X_test, Y_test)
print("K Nearest Neighbors Accuracy:", accuracy_knn)

# Make predictions on a new data point
# new_data_point = sc.transform(np.array([[86, 66, 26.6, 31]]))
# prediction = knn.predict(new_data_point)
# print("Prediction for new data point:", prediction)

# Save the KNN model
pickle.dump(knn, open('model_knn.pkl', 'wb'))


