#!/usr/bin/env python
# coding: utf-8

# # Import Data 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read a CSV file
data_train = pd.read_csv('adult.csv')
column_names = ['age', 'workclass', 'fnlwgt', 'education','education_num', 'marital-status', 
                'occupation', 'relationship', 'race', 'sex', 'Capital-gain', 
                'Capital-loss', 'hours-per-week', 'Native-country', 'value']
data_train.columns = column_names
data_train.head()
print(data_train.shape)


# In[2]:


data_test = pd.read_csv('adult_test.csv')
data_test.columns= column_names
data_test.head()


# ### Combine both Train and Test Data

# In[3]:


complete_data = data_train.append(data_test)
complete_data.head()


# # DATA PROCESSING

# ### Replace ? with null value and remove null value 

# In[4]:


complete_data.replace(' ?', np.nan, inplace=True)
complete_data = complete_data.dropna()
missing_data = complete_data.isnull().sum()
print(complete_data.info())


# ### Transform Categorical data to encoder

# In[5]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
workclass01 = le.fit_transform(complete_data['workclass'])

complete_data.insert(1,"workclass01", workclass01, True)


# In[6]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
maritalstatus01 = le.fit_transform(complete_data['marital-status'])

complete_data.insert(1,"marital-status01", maritalstatus01, True)


# In[7]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
occupation01 = le.fit_transform(complete_data['occupation'])

complete_data.insert(1,"occupation01", occupation01, True)


# In[8]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
relationship01 = le.fit_transform(complete_data['relationship'])

complete_data.insert(1,"relationship01", relationship01, True)


# In[9]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
race01 = le.fit_transform(complete_data['race'])

complete_data.insert(1,"race01", race01, True)


# In[10]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
sex01 = le.fit_transform(complete_data['sex'])

complete_data.insert(1,"sex01", sex01, True)


# In[11]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
Nativecountry01 = le.fit_transform(complete_data['Native-country'])

complete_data.insert(1,"Native-country01", Nativecountry01, True)


# In[12]:


from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()
value01 = le.fit_transform(complete_data['value'])

complete_data.insert(1,"value01", value01, True)


# ### Create new table for encoded variables

# In[14]:


complete_data_categorical = complete_data.drop(['age', 'workclass', 'fnlwgt', 'education','education_num', 'marital-status', 
                'occupation', 'relationship', 'race', 'sex', 'Capital-gain', 
                'Capital-loss', 'hours-per-week', 'Native-country', 'value'], axis =1)


# In[15]:


complete_data_categorical.head()


# In[18]:


complete_data_new = complete_data.drop([ 'workclass', 'education', 'marital-status', 
                'occupation', 'relationship', 'race', 'sex', 'Native-country', 'value'], axis =1)


# # Correlation check 

# In[16]:


complete_data.corr()['value01'].sort_values(ascending=False)


# In[20]:


complete_data_new02 = complete_data_new.drop(['fnlwgt', 'relationship01', 'marital-status01'], axis =1)
complete_data_new02.head()


# ### standardized numerical data

# In[24]:


from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
x_stan = complete_data_new02.drop('value01', axis =1)
y= complete_data_new02['value01']
x_stan_train,x_stan_test,y_train,y_test = train_test_split(x_stan,y,test_size=0.3,random_state=20)

numerical_columns = ['age','education_num', 'Capital-gain', 'Capital-loss', 'hours-per-week']

ss =StandardScaler(with_mean=False)

x_stan_train[numerical_columns] = ss.fit_transform(x_stan_train[numerical_columns])
x_stan_test[numerical_columns] =ss.fit_transform(x_stan_test[numerical_columns])


# # 1st Model SVM 

# ### SVM - Linear

# In[65]:


from sklearn import svm
import time
starttime= time.time()
linear = svm.SVC(kernel='linear')
linear.fit(x_stan_train,y_train)
trainingtime= time.time()-starttime
print(trainingtime)


# In[66]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

prediction = linear.predict(x_stan_test)  # Predict on test data
    
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average='weighted')
recall = recall_score(y_test, prediction, average='weighted')
f1 = f1_score(y_test, prediction, average='weighted')
confusion = confusion_matrix(y_test, prediction)

print(accuracy)
print(precision)
print(recall)
print(f1)
print(confusion)


# ### SVM - Radial 

# In[67]:


from sklearn import svm
import time
starttime= time.time()
rbf = svm.SVC(kernel='rbf')
rbf.fit(x_stan_train,y_train)
trainingtime= time.time()-starttime
print(trainingtime)


# In[68]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

prediction = rbf.predict(x_stan_test)  # Predict on test data
    
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average='weighted')
recall = recall_score(y_test, prediction, average='weighted')
f1 = f1_score(y_test, prediction, average='weighted')
confusion = confusion_matrix(y_test, prediction)

print(accuracy)
print(precision)
print(recall)
print(f1)
print(confusion)


# ### SVM - Poly

# In[69]:


from sklearn import svm
import time
starttime= time.time()
degree = [2]

for i in degree:
    poly = svm.SVC(C=10, kernel='poly', degree=i, gamma='scale')
    poly.fit(x_stan_train,y_train)
trainingtime= time.time()-starttime
print(trainingtime)
prediction = poly.predict(x_stan_test)  # Predict on test data
    
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average='weighted')
recall = recall_score(y_test, prediction, average='weighted')
f1 = f1_score(y_test, prediction, average='weighted')
confusion = confusion_matrix(y_test, prediction)

print(accuracy)
print(precision)
print(recall)
print(f1)
print(confusion)


# ### Hyperparameter tuning for Poly-5

# In[70]:


starttime= time.time()
degree = [2,3,4,5]

for i in degree:
    starttime= time.time()
    poly = svm.SVC(C=10, kernel='poly', degree=i, gamma='scale')
    poly.fit(x_stan_train,y_train)
    trainingtime= time.time()-starttime
    print(i)
    print(trainingtime)
    prediction = poly.predict(x_stan_test)  # Predict on test data
    
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    f1 = f1_score(y_test, prediction, average='weighted')
    confusion = confusion_matrix(y_test, prediction)

    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    print(confusion)


# In[ ]:


c = [1,10,100]

for i in c:
    starttime= time.time()
    poly = svm.SVC(C=i, kernel='poly', degree=5, gamma='scale', probability=True,decision_function_shape='ovr')
    poly.fit(x_stan_train,y_train)
    trainingtime= time.time()-starttime
    print(i)
    print(trainingtime)
    prediction = poly.predict(x_stan_test)  # Predict on test data
    
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    f1 = f1_score(y_test, prediction, average='weighted')
    confusion = confusion_matrix(y_test, prediction)

    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    print(confusion)


# # Random Tree 
# 

# In[64]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

starttime= time.time()
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(x_stan_train, y_train)

# Make predictions
y_pred = clf.predict(x_stan_test)

trainingtime= time.time()-starttime
print(trainingtime)

# Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ### Search for the best hyperparameter for random tree

# In[63]:


from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestClassifier(random_state=42)

# Set up the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(x_stan_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)


# In[73]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

starttime= time.time()
best_params = {
    'max_depth': 10,
    'max_features': 'auto',
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 100
}

rf_model = RandomForestClassifier(**best_params, random_state=42)

# Train the model
rf_model.fit(x_stan_train, y_train)

# Make predictions
y_pred = rf_model.predict(x_stan_test)

trainingtime= time.time() - starttime
print(trainingtime)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optionally, calculate and print accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




