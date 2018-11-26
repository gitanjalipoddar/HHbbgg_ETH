
# coding: utf-8

# In[2]:

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterSampler
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler


# In[3]:

X=pd.read_hdf("X_train.hd5")
y=pd.read_hdf("y_train.hd5")
w=pd.read_hdf("w_train.hd5")
classifier = joblib.load('clf.joblib')


# In[4]:

#parameter grid
param_grid = {
              "clip_weight" : [10,20,30,40],
              "learning_rate" : [0.001,0.1,0.5],
              "n_estimators": [300,500,800],
              "subsample" : [0.6, 0.8, 1],
              "reg_lambda":[0.1, 0.5, 1, 2, 10],  
              "max_depth": [3,5,7,10],                                                                                                                                                                                                
              }


# In[5]:

#60 parameter samples
sampler=ParameterSampler(param_grid,60)
samples=[params for params in sampler]

#array to store accuracy scores 
accu_scores=np.array([]) 
#array to store cross-entropy
cross_scores=np.array([])

scaler=RobustScaler()


# In[14]:

for params in sampler:
    
    skf = StratifiedKFold(n_splits=5)
    clipweight=params.pop('clip_weight')
    for train_index, test_index in skf.split(X, y):
        
        #obtaining training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        w_train, w_test = w.iloc[train_index], w.iloc[test_index]
        
        #scale data
        X_train=pd.DataFrame(scaler.fit_transform(X_train))
        X_test=pd.DataFrame(scaler.transform(X_test))
        
        #reweighting w_train with clipped weights
        h=np.histogram(y_train,weights=w_train,bins=13,range=[-0.5,12.5])
        a=1./h[0]
        a/=min(a)
        rw=np.clip(a,0,clipweight)
        w_train*=rw[y_train]
        
        #classifier with XGBoost parameters and training it
        clf=deepcopy(classifier)
        clf.set_params(**params)
        clf.fit(X_train,y_train,w_train)
        
        #ignoring category 0 from y_true (and hence from X_true and w_true too)
        X_test.reset_index(drop=True,inplace=True)
        y_test.reset_index(drop=True,inplace=True)
        w_test.reset_index(drop=True,inplace=True)
        X_test_ignore0=X_test[y_test>0]
        y_test_ignore0=y_test[y_test>0]
        w_test_ignore0=w_test[y_test>0]
        
        #getting predicted y probability, ignoring category 0
        y_pred_prob=clf.predict_proba(X_test_ignore0) 
        y_pred_prob/=rw.reshape(1,-1)
        y_pred_prob/=np.sum(y_pred_prob,axis=1,keepdims=True)
        
        y_pred_prob_ignore0=np.delete(y_pred_prob,0,axis=1)
        y_pred_prob_ignore0/=np.sum(y_pred_prob_ignore0,axis=1,keepdims=True)
        
        #getting sample weight values, ignoring category 0
        weight=w_test_ignore0.ravel()
        
        #calculating accuracy score
        y_pred_ignore0=np.argmax(y_pred_prob_ignore0,axis=1)+1
        y_pred=y_pred_ignore0.ravel()
        y_true=y_test_ignore0.ravel()
        accu_scores=np.append(accu_scores,accuracy_score(y_true,y_pred,normalize=True,sample_weight=weight))
        
        #calculation cross entropy score
        enc=OneHotEncoder(handle_unknown='ignore')
        y_label=enc.fit_transform(y_test_ignore0.reshape(-1,1)).toarray()
        cross_scores=np.append(cross_scores,
                               log_loss(y_label,y_pred_prob_ignore0,normalize=True,sample_weight=w_test_ignore0))


# In[18]:

scores1=np.split(accu_scores,60)
np.savetxt('accu_scores.txt',scores1)
scores2=np.split(cross_scores,60)
np.savetxt('cross_scores.txt',scores2)


# In[6]:

#arrays to store mean values of accuracy and cross entropy scores
accu_mean=np.array([])
cross_mean=np.array([])

f=open('optimizationscores.txt', 'w')
for i in range (0,len(scores1)):
    f.write("Parameters: %s\n"%samples[i])
    f.write("Accuracy with w_test: %0.5f +/- %0.5f\n"%(scores1[i].mean(),scores1[i].std()))
    f.write("Cross entropy: %0.5f +/- %0.5f\n"%(scores2[i].mean(),scores2[i].std()))
    f.write("\n")
    
    accu_mean=np.append(accu_mean, scores1[i].mean())
    cross_mean=np.append(cross_mean, scores2[i].mean())
f.close()

#sorting arrays with mean values
cross_mean.sort()
accu_mean.sort()
np.savetxt('accu_mean.txt',accu_mean)
np.savetxt('cross_mean.txt',cross_mean)
# In[ ]:



