#!/usr/bin/env python
# coding: utf-8

# Master thesis
# 
# 
# Maartje Verhoeven | ANR: 706805 | SNR: u1273860 
# 
# 

# In[1]:


#packages
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[4]:


#F3D
phone_mood = pd.read_csv ('first3_days.csv', index_col = 0)
phone_mood = pd.DataFrame(phone_mood)
phone_mood = phone_mood[phone_mood["mood"].notnull()]

#Label encounter
le = preprocessing.LabelEncoder()
phone_mood["day_part"] = le.fit_transform(phone_mood["day_part"].astype("str"))
phone_mood["mood"] = le.fit_transform(phone_mood["mood"].astype("str"))

#Creating X
X = phone_mood.drop(["user_id", "date", "mood"], axis = 1)
X = X.astype("int")
X = np.array(X)

#Creating Y
y = np.array(phone_mood["mood"])
y = y.astype("int")

#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3) 

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)

X_train = scaling.transform(X_train)
X_val = scaling.transform(X_val)

from collections import Counter
print("counts in y_train")
print(Counter(y_train))

print("counts in y_val")
print(Counter(y_val))

#Gridsearch RandomForests
from sklearn.ensemble import RandomForestClassifier

n_estimators = list(range(10,101,10))
max_depth = list(range(2, 20, 2))

param_grid = {"n_estimators": n_estimators, "max_depth" : max_depth}

rf = RandomForestClassifier()
rf_search = GridSearchCV(rf, param_grid, cv = 5)
rf_search.fit(X_train,y_train)

print("Best parameters: ", rf_search.best_params_)
print("Best accuracy score:", rf_search.best_score_)

#Random Forest Confusion Matrix
rf_search_predictions = rf_search.predict(X_val)
print(classification_report(y_val, rf_search_predictions))

import itertools
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Pleasant Activation','Pleasant Dectivation',
               'Unpleasant Activation', 'Unpleasant Dectivation']

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlim(-0.5, len(np.unique(y))-0.5)
    plt.ylim(len(np.unique(y))-0.5, -0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, rf_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM F3D.png', bbox_inches='tight')

plt.show()


# In[5]:


# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM F3D.png', bbox_inches='tight')

plt.show()


# In[23]:


#majority baseline F3D y_train
104/191 #0.544502


# In[7]:


#majority baseline F3D y_val
45/82 #0.54878


# In[8]:


#L3D
phone_mood = pd.read_csv ('last3_days.csv', index_col = 0)
phone_mood = pd.DataFrame(phone_mood)
phone_mood = phone_mood[phone_mood["mood"].notnull()]

#Label encounter
le = preprocessing.LabelEncoder()
phone_mood["day_part"] = le.fit_transform(phone_mood["day_part"].astype("str"))
phone_mood["mood"] = le.fit_transform(phone_mood["mood"].astype("str"))

#Creating X
X = phone_mood.drop(["user_id", "date", "mood"], axis = 1)
X = X.astype("int")
X = np.array(X)

#Creating Y
y = np.array(phone_mood["mood"])
y = y.astype("int")

#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3) 

X_train = scaling.transform(X_train)
X_val = scaling.transform(X_val)

from collections import Counter
print("counts in y_train")
print(Counter(y_train))

print("counts in y_val")
print(Counter(y_val))

#Gridsearch RandomForests
from sklearn.ensemble import RandomForestClassifier

n_estimators = list(range(10,101,10))
max_depth = list(range(2, 20, 2))

param_grid = {"n_estimators": n_estimators, "max_depth" : max_depth}

rf = RandomForestClassifier()
rf_search = GridSearchCV(rf, param_grid, cv = 5)
rf_search.fit(X_train,y_train)

print("Best parameters: ", rf_search.best_params_)
print("Best accuracy score:", rf_search.best_score_)

#Random Forest Confusion Matrix
rf_search_predictions = rf_search.predict(X_val)
print(classification_report(y_val, rf_search_predictions))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, rf_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM L3D.png', bbox_inches='tight')

plt.show()


# In[9]:


#majority baseline L3D y_train
23/52 


# In[10]:


##majority baseline L3D y_val
13/23


# In[11]:


#FH
phone_mood = pd.read_csv ('first_half.csv', index_col = 0)
phone_mood = pd.DataFrame(phone_mood)
phone_mood = phone_mood[phone_mood["mood"].notnull()]

#Label encounter
le = preprocessing.LabelEncoder()
phone_mood["day_part"] = le.fit_transform(phone_mood["day_part"].astype("str"))
phone_mood["mood"] = le.fit_transform(phone_mood["mood"].astype("str"))

#Creating X
X = phone_mood.drop(["user_id", "date", "mood"], axis = 1)
X = X.astype("int")
X = np.array(X)

#Creating Y
y = np.array(phone_mood["mood"])
y = y.astype("int")

#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3) 

X_train = scaling.transform(X_train)
X_val = scaling.transform(X_val)

from collections import Counter
print("counts in y_train")
print(Counter(y_train))

print("counts in y_val")
print(Counter(y_val))

#Gridsearch RandomForests
from sklearn.ensemble import RandomForestClassifier

n_estimators = list(range(10,101,10))
max_depth = list(range(2, 20, 2))

param_grid = {"n_estimators": n_estimators, "max_depth" : max_depth}

rf = RandomForestClassifier()
rf_search = GridSearchCV(rf, param_grid, cv = 5)
rf_search.fit(X_train,y_train)

print("Best parameters: ", rf_search.best_params_)
print("Best accuracy score:", rf_search.best_score_)

#Random Forest Confusion Matrix
rf_search_predictions = rf_search.predict(X_val)
print(classification_report(y_val, rf_search_predictions))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, rf_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM FH.png', bbox_inches='tight')

plt.show()


# In[12]:


#majority baseline FH y_train
1025/2065 


# In[13]:


#majority baseline FH y_val
434/885 


# In[20]:


#LH
phone_mood = pd.read_csv ('second_half.csv', index_col = 0)
phone_mood = pd.DataFrame(phone_mood)
phone_mood = phone_mood[phone_mood["mood"].notnull()]

#Label encounter
le = preprocessing.LabelEncoder()
phone_mood["day_part"] = le.fit_transform(phone_mood["day_part"].astype("str"))
phone_mood["mood"] = le.fit_transform(phone_mood["mood"].astype("str"))

#Creating X
X = phone_mood.drop(["user_id", "date", "mood"], axis = 1)
X = X.astype("int")
X = np.array(X)

#Creating Y
y = np.array(phone_mood["mood"])
y = y.astype("int")

#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3) 

X_train = scaling.transform(X_train)
X_val = scaling.transform(X_val)

from collections import Counter
print("counts in y_train")
print(Counter(y_train))

print("counts in y_val")
print(Counter(y_val))

#Gridsearch RandomForests
from sklearn.ensemble import RandomForestClassifier

n_estimators = list(range(10,101,10))
max_depth = list(range(2, 20, 2))

param_grid = {"n_estimators": n_estimators, "max_depth" : max_depth}

rf = RandomForestClassifier()
rf_search = GridSearchCV(rf, param_grid, cv = 5)
rf_search.fit(X_train,y_train)

print("Best parameters: ", rf_search.best_params_)
print("Best accuracy score:", rf_search.best_score_)

#Random Forest Confusion Matrix
rf_search_predictions = rf_search.predict(X_val)
print(classification_report(y_val, rf_search_predictions))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, rf_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM LH.png', bbox_inches='tight')

plt.show()


# In[21]:


#majority baseline LH y_train
748/1639


# In[22]:


#majority baseline LH y_val
330/703

