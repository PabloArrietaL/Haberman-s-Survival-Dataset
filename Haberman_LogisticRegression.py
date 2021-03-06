#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sklearn.metrics as m



# In[2]:


def read_split():

    columns = ['Age','Year','Nodes detected','Status']
    dataset = pd.read_csv("haberman.txt",header=None,names=columns)
    dataset['Status'] = dataset['Status'].apply({2:0,1:1}.get)
    X = dataset.drop(labels=['Status'],axis=1).values.astype(float)
    y = dataset.Status.values
    X_t,X_te, y_t, y_te = train_test_split(X,y,test_size=0.30, random_state=0, stratify=y)
    X_train_Nor = StandardScaler().fit_transform(X_t[:])
    X_test_Nor = StandardScaler().fit_transform(X_te[:])
    return(X_train_Nor,y_t,X_test_Nor,y_te)


# In[3]:


def metrics(test,model,c):
    f1 = m.f1_score(test,model)
    recall = m.recall_score(test,model)
    accuracy = m.accuracy_score(test,model)
    precision = m.precision_score(test,model)
    tn, fp, fn, tp = m.confusion_matrix(test,model).ravel()
    specificity = tn/float(tn+fp)
    print("\n\t   CONFUSION MATRIX")
    print("         Negative     Positive")
    print("Negative   {0}           {1}".format(tn,fp))
    print("Positive   {0}            {1}".format(fn,tp))
    print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1,recall,accuracy,precision,specificity))
    print_metrics(f1,recall,accuracy,precision,specificity,tn, fp, fn, tp, c)


# In[4]:


def print_metrics(f1,recall,accuracy,precision,specificity,tn, fp, fn, tp, c):
    f = open('logistic.txt','w')
    f.write("F1: "+str(f1)+'\n')
    f.write("Recall: "+str(recall)+'\n')
    f.write("Accuracy: "+str(accuracy)+'\n')
    f.write("Precision: "+str(precision)+'\n')
    f.write("Specificity "+str(specificity)+'\n')
    f.write("TN: "+str(tn)+'\n')
    f.write("FP: "+str(fp)+'\n')
    f.write("FN: "+str(fn)+'\n')
    f.write("TP: "+str(tp)+'\n')
    f.write("C: "+str(c)+'\n')
    f.close()


# In[5]:


if __name__ == "__main__":
    X_train,y_train,X_test,y_test = read_split()
    logistic = LogisticRegression(solver="lbfgs", max_iter=500)
    C = [10,100,1.0,0.1,0.01,0.001,2.0,3.0,0.0001,0.0015]
    hyperparameters = dict(C=C)
    clf = GridSearchCV(logistic, hyperparameters, cv=10)
    bs_model = clf.fit(X_train,y_train)
    bs_C = bs_model.best_estimator_.get_params(['C'])
    Logis_model = LogisticRegression(C=bs_C['C'],solver="lbfgs", max_iter=500)
    Logis_model.fit(X_train,y_train)
    model = Logis_model.predict(X_test)
    print("\tLogistic Regression metrics")
    metrics(y_test,model,bs_C['C'])



