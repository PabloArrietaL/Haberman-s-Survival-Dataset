#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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


def metrics(test,model,alpha,hls):
    
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
    print_metrics(f1,recall,accuracy,precision,specificity,tn, fp, fn, tp, alpha, hls)


# In[4]:


def print_metrics(f1,recall,accuracy,precision,specificity,tn, fp, fn, tp, alpha, hls):
    f = open('MLP.txt','w')
    f.write("F1: "+str(f1)+'\n')
    f.write("Recall: "+str(recall)+'\n')
    f.write("Accuracy: "+str(accuracy)+'\n')
    f.write("Precision: "+str(precision)+'\n')
    f.write("Specificity "+str(specificity)+'\n')
    f.write("TN: "+str(tn)+'\n')
    f.write("FP: "+str(fp)+'\n')
    f.write("FN: "+str(fn)+'\n')
    f.write("TP: "+str(tp)+'\n')
    f.write("Alpha: "+str(alpha)+'\n')
    f.write("Hidden Layer: "+str(hls)+'\n')
    f.close()


# In[5]:


def neurons(fe,outs,sam,alpha):
    N = sam/(alpha*(fe+outs))
    return N


# In[6 ]:

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    X_train,y_train,X_test,y_test = read_split()
    s = len(list(X_train))
    f = len(list(X_train[0]))
    mlp = MLPClassifier(max_iter=2000)
    alpha = 10.0 ** -np.arange(1, 7)
    hidden = [round((neurons(f,1,s,2)),),(round((neurons(f,1,s,2))),round((neurons(f,1,s,2)))),(round(neurons(f,1,s,5)),),(round(neurons(f,1,s,10)),),
    (round(neurons(f,1,s,5)),round(neurons(f,1,s,5)))]
    hyperparameters = dict(alpha=alpha, hidden_layer_sizes=hidden)
    clf = GridSearchCV(mlp,hyperparameters, cv=10)
    bs_model = clf.fit(X_train,y_train)
    bs_alpha = bs_model.best_estimator_.get_params(['alpha'])
    bs_hls = bs_model.best_estimator_.get_params(['hidden_layer_sizes'])
    mlp_model = MLPClassifier(alpha=bs_alpha['alpha'], hidden_layer_sizes=bs_hls['hidden_layer_sizes'], max_iter=2000)
    mlp_model.fit(X_train,y_train)
    model = mlp_model.predict(X_test)
    print("\tMLP metrics")
    metrics(y_test,model,bs_alpha['alpha'],bs_hls['hidden_layer_sizes'])

