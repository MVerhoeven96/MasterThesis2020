#!/usr/bin/env python
# coding: utf-8

# Master thesis
# 
# 
# Maartje Verhoeven | ANR: 706805 | SNR: u1273860 
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk

print(pd.__version__)
print(np.__version__)
print(sk.__version__)


# In[2]:


#packages
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


# In[2]:


phone_mood = pd.read_csv ('final.csv', index_col = 0)
phone_mood = pd.DataFrame(phone_mood)
phone_mood = phone_mood[phone_mood["mood"].notnull()]

# In[3]:


#Unscaled RF

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

class_names = ['Pleasant Deactivation','Pleasant Activation',
               'Unpleasant Deactivation', 'Unpleasant Activation']

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

plt.savefig('CM RF.png', bbox_inches='tight')

plt.show()


# In[7]:


# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM RF.png', bbox_inches='tight')

plt.show()


# In[3]:


#scaled SVM, LR, KNN, RF
phone_mood = pd.read_csv ('final.csv', index_col = 0)
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

# In[7]:


from sklearn import svm

kernel = ["rbf", "linear", "sigmoid"]
gamma = ["scale", "auto"]
decision_function_shape = ["ovo", "ovr"]
param_grid = {"kernel": kernel, "gamma" : gamma, "decision_function_shape" : decision_function_shape}

clf = svm.SVC()
svm_search = GridSearchCV(clf, param_grid, cv = 5)
svm_search.fit(X_train, y_train)

print("Best parameters:", svm_search.best_params_)
print("Best accuracy score:", svm_search.best_score_)

#SVM Confusionmatrix
svm_search_predictions = svm_search.predict(X_val)
print(confusion_matrix(y_val, svm_search_predictions))
print(classification_report(y_val, svm_search_predictions))

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
cnf_matrix = confusion_matrix(y_val, svm_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM CVM.png', bbox_inches='tight')

plt.show()


# In[8]:


# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM CVM.png', bbox_inches='tight')

plt.show()


# In[9]:


#Gridsearch Logistic Regression
from sklearn.linear_model import LogisticRegression

penalty = ["l1","l2"]
C = np.logspace(-3,3,7)

param_grid = {"C": C, "penalty": penalty}

logreg = LogisticRegression(multi_class = "auto") #solver = lbfgs is needed for L1 but not for l2
logreg_search = GridSearchCV(logreg, param_grid, cv = 5)
logreg_search.fit(X_train, y_train)

print("Best parameters: ",logreg_search.best_params_)
print("Best accuracy score:", logreg_search.best_score_)

#logistic regression Confusion Matrix
logreg_search_predictions = logreg_search.predict(X_val)
print(confusion_matrix(y_val, logreg_search_predictions))
print(classification_report(y_val, logreg_search_predictions))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, logreg_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM LR.png', bbox_inches='tight')

plt.show()


# In[10]:


#Gridsearch KNN
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = list(range(1,25))
weights = ["uniform", "distance"]

param_grid = {"n_neighbors": n_neighbors, "weights": weights}

knn = KNeighborsClassifier()
knn_search = GridSearchCV(knn, param_grid, cv = 5)
knn_search.fit(X_train,y_train)

print("Best parameters: ", knn_search.best_params_)
print("Best accuracy score:", knn_search.best_score_)

#KNN Confusion Matrix
knn_search_predictions = knn_search.predict(X_val)
print(confusion_matrix(y_val, knn_search_predictions ))
print(classification_report(y_val, knn_search_predictions ))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, knn_search_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (8, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.tight_layout()

plt.savefig('CM KNN.png', bbox_inches='tight')

plt.show()


# In[11]:


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

plt.savefig('CM RF.png', bbox_inches='tight')

plt.show()

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

plt.savefig('CM RF.png', bbox_inches='tight')

plt.show()

