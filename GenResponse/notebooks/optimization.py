
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
from sklearn.model_selection import train_test_split


# In[3]:

features = ['hh_m','hh_pt', 'hh_eta','hgg_pt_hh_m','hbb_pt_hh_m','cos_theta_cs','cos_theta_hbb','cos_theta_hgg']
features+=['leadJet_pt','leadJet_eta','subleadJet_pt','subleadJet_eta']
features+=['leadPho_pt','leadPho_eta','subleadPho_pt','subleadPho_eta']

scaler=RobustScaler()

node_6=pd.read_hdf("node_6.hd5")
node_9=pd.read_hdf("node_9.hd5")
node_4=pd.read_hdf("node_4.hd5")
node_SM=pd.read_hdf("node_SM.hd5")

frames=[node_6,node_9,node_4,node_SM]
df=pd.concat(frames)

df=df[df.cat>0]
df.weight/=df.weight.mean()

#indexing
random_index = np.arange(df.shape[0]) 
np.random.shuffle(random_index)
df["random_index"]=random_index 
df.set_index("random_index",inplace=True)
df.sort_index(inplace=True)

X = df[features]
y = df['cat'] 
w = df['weight']

w=np.abs(w)
classifier= joblib.load('clf.joblib') 


# In[4]:

#parameter grid
param_grid = {
              "clip_weight" : [0.01,0.05,0.1,0.5],
              "learning_rate" : [0.1,0.3,0.5],
              "n_estimators": [300,500,800],
              "subsample" : [0.6, 0.8, 1],
              "reg_lambda":[0.1, 0.5, 1, 2, 10],  
              "max_depth": [3,5,7,10],                                                                                                                                                                                                
              }


# In[5]:

#60 parameter samples
sampler=ParameterSampler(param_grid,60)
samples=[params for params in sampler]
df=pd.DataFrame(samples)

#array to store accuracy scores 
accu_scores=np.array([]) 
accu_mean=np.array([]) 
accu_stdev=np.array([]) 
#array to store cross-entropy
cross_scores=np.array([])
cross_mean=np.array([]) 
cross_stdev=np.array([])


# In[14]:

for params in samples:
    
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
        bins=np.arange(0.5,13.5,step=1)
        h=np.histogram(y_train,weights=w_train,bins=bins)
        a=1./h[0]
        a/=min(a)
        rw=np.clip(a,0,clipweight)
        w_train*=rw[y_train-1]
        
        #classifier with XGBoost parameters and training it
        clf=deepcopy(classifier)
        clf.set_params(**params)
        clf.fit(X_train,y_train,w_train)
        
        #getting predicted y probability
        y_pred_prob=clf.predict_proba(X_test) 
        y_pred_prob/=rw.reshape(1,-1)
        y_pred_prob/=np.sum(y_pred_prob,axis=1,keepdims=True)
       
        #calculating accuracy score
        y_pred=np.argmax(y_pred_prob,axis=1)+1
        y_true=y_test.ravel()
        accu_scores=np.append(accu_scores,accuracy_score(y_true,y_pred.ravel(),normalize=True,sample_weight=w_test))
        
        #calculation cross entropy score
        enc=OneHotEncoder(handle_unknown='ignore')
        y_label=enc.fit_transform(y_test.reshape(-1,1)).toarray()
        cross_scores=np.append(cross_scores,
                               log_loss(y_label,y_pred_prob,normalize=True,sample_weight=w_test))


# In[18]:

scores1=np.split(accu_scores,60)
scores2=np.split(cross_scores,60)

for i in range (0,len(scores1)):
    accu_mean=np.append(accu_mean,scores1[i].mean())
    accu_stdev=np.append(accu_stdev,scores1[i].std())
    cross_mean=np.append(cross_mean,scores2[i].mean())
    cross_stdev=np.append(cross_stdev,scores2[i].std())

#adding mean and stdev
df['accu_mean']=accu_mean
df['accu_stdev']=accu_stdev
df['cross_mean']=cross_mean
df['cross_stdev']=cross_stdev

#sorting accuracy in descending order
df_accu=df.sort_values('accu_mean',ascending=False,inplace=False)
#sorting cross entropy in ascending order
df_cross=df.sort_values('cross_mean',ascending=True,inplace=False)

#saving outputs
df.to_hdf('optimisation.hd5', key='df', mode='w')
df_accu.to_hdf('optimisation_accu.hd5', key='df', mode='w')
df_cross.to_hdf('optimisation_cross.hd5', key='df', mode='w')
# In[ ]:



