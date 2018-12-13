
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
node_4=pd.read_hdf("node_4.hd5")
node_SM=pd.read_hdf("node_SM.hd5")
node_9=pd.read_hdf("node_9.hd5")

frames=[node_6,node_4,node_SM,node_9]
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

#removing category 0 and changing weights for individual nodes
node_6=node_6[node_6.cat>0]
node_6.weight/=node_6.weight.mean()
    
node_4=node_4[node_4.cat>0]
node_4.weight/=node_4.weight.mean()

node_SM=node_SM[node_SM.cat>0]
node_SM.weight/=node_SM.weight.mean()

node_9=node_9[node_9.cat>0]
node_9.weight/=node_9.weight.mean()

nodes=[node_4,node_6,node_9,node_SM]

# In[4]:

#parameter grid
param_grid = {
              "clip_weight" : [10,20,30,40],
              "min_child_weight": [1000,200,100,20],
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
result=pd.DataFrame(samples)

#array to store accuracy scores 
accu_scores=np.array([]) 
accu_scores_nodes=np.array([])
accu_mean=np.array([]) 
accu_stdev=np.array([]) 
accu_mean_4=np.array([]) 
accu_stdev_4=np.array([]) 
accu_mean_6=np.array([]) 
accu_stdev_6=np.array([]) 
accu_mean_9=np.array([]) 
accu_stdev_9=np.array([])
accu_mean_SM=np.array([]) 
accu_stdev_SM=np.array([])
#array to store cross-entropy
cross_scores=np.array([])
cross_scores_nodes=np.array([])
cross_mean=np.array([]) 
cross_stdev=np.array([])
cross_mean_4=np.array([]) 
cross_stdev_4=np.array([])
cross_mean_6=np.array([]) 
cross_stdev_6=np.array([])
cross_mean_9=np.array([]) 
cross_stdev_9=np.array([])
cross_mean_SM=np.array([]) 
cross_stdev_SM=np.array([])

#function to calculate scores for nodes
def nodes_scores(node,clf,rw):
    X = node[features]
    y = node['cat'] 
    w = node['weight']

    #slicing data randomly into training and testing sets- we take 20% to be the testing set
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w,test_size=0.2)
    w_train = np.abs(w_train)

    #scale data
    X_train=pd.DataFrame(scaler.fit_transform(X_train))
    X_test=pd.DataFrame(scaler.transform(X_test))

    y_pred_prob=clf.predict_proba(X_test)
    y_pred_prob/=rw.reshape(1,-1) 
    y_pred_prob/=np.sum(y_pred_prob,axis=1,keepdims=True)
    
    #calculating accuracy score
    y_pred=np.argmax(y_pred_prob,axis=1)+1
    y_true=y_test.ravel()
    accu=accuracy_score(y_true,y_pred.ravel(),normalize=True,sample_weight=w_test)
        
    #calculation cross entropy score
    enc=OneHotEncoder(handle_unknown='ignore')
    y_label=enc.fit_transform(y_test.reshape(-1,1)).toarray()
    cross=log_loss(y_label,y_pred_prob,normalize=True,sample_weight=w_test)
    
    return accu, cross

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
        
        #calculating scores for nodes
        for i in range (0,4):
            node=nodes[i]
            accu,cross=nodes_scores(node,clf,rw) 
            accu_scores_nodes=np.append(accu_scores_nodes,accu)
            cross_scores_nodes=np.append(cross_scores_nodes,cross)


# In[18]:

#splitting into parameter sets
scores1=np.split(accu_scores,60)
scores2=np.split(cross_scores,60)

for i in range (0,len(scores1)):
    accu_mean=np.append(accu_mean,scores1[i].mean())
    accu_stdev=np.append(accu_stdev,scores1[i].std())
    cross_mean=np.append(cross_mean,scores2[i].mean())
    cross_stdev=np.append(cross_stdev,scores2[i].std())

#splitting into parameter sets
scores1=np.split(accu_scores_nodes,60)
scores2=np.split(cross_scores_nodes,60)

for i in range (0,len(scores1)):
    #splitting into accuracy scores per node and calculating mean, std
    array=scores1[i]
    
    accu_4=np.array([array[0],array[4],array[8],array[12],array[16]])
    accu_mean_4=np.append(accu_mean_4,accu_4.mean())
    accu_stdev_4=np.append(accu_stdev_4,accu_4.std())
    
    accu_6=np.array([array[1],array[5],array[9],array[13],array[17]])
    accu_mean_6=np.append(accu_mean_6,accu_6.mean())
    accu_stdev_6=np.append(accu_stdev_6,accu_6.std())
    
    accu_9=np.array([array[2],array[6],array[10],array[14],array[18]])
    accu_mean_9=np.append(accu_mean_9,accu_9.mean())
    accu_stdev_9=np.append(accu_stdev_9,accu_9.std())
    
    accu_SM=np.array([array[3],array[7],array[11],array[15],array[19]])
    accu_mean_SM=np.append(accu_mean_SM,accu_SM.mean())
    accu_stdev_SM=np.append(accu_stdev_SM,accu_SM.std())
    
    #splitting into entropy scores per node and calculating mean, std
    array=scores2[i]
    
    cross_4=np.array([array[0],array[4],array[8],array[12],array[16]])
    cross_mean_4=np.append(cross_mean_4,cross_4.mean())
    cross_stdev_4=np.append(cross_stdev_4,cross_4.std())
    
    cross_6=np.array([array[1],array[5],array[9],array[13],array[17]])
    cross_mean_6=np.append(cross_mean_6,cross_6.mean())
    cross_stdev_6=np.append(cross_stdev_6,cross_6.std())
    
    cross_9=np.array([array[2],array[6],array[10],array[14],array[18]])
    cross_mean_9=np.append(cross_mean_9,cross_9.mean())
    cross_stdev_9=np.append(cross_stdev_9,cross_9.std())
    
    cross_SM=np.array([array[3],array[7],array[11],array[15],array[19]])
    cross_mean_SM=np.append(cross_mean_SM,cross_SM.mean())
    cross_stdev_SM=np.append(cross_stdev_SM,cross_SM.std())

#adding mean and stdev
result['accu_mean']=accu_mean
result['accu_stdev']=accu_stdev
result['cross_mean']=cross_mean
result['cross_stdev']=cross_stdev

result['accu_mean_4']=accu_mean_4
result['accu_stdev_4']=accu_stdev_4
result['cross_mean_4']=cross_mean_4
result['cross_stdev_4']=cross_stdev_4

result['accu_mean_6']=accu_mean_6
result['accu_stdev_6']=accu_stdev_6
result['cross_mean_6']=cross_mean_6
result['cross_stdev_6']=cross_stdev_6

result['accu_mean_9']=accu_mean_9
result['accu_stdev_9']=accu_stdev_9
result['cross_mean_9']=cross_mean_9
result['cross_stdev_9']=cross_stdev_9

result['accu_mean_SM']=accu_mean_SM
result['accu_stdev_SM']=accu_stdev_SM
result['cross_mean_SM']=cross_mean_SM
result['cross_stdev_SM']=cross_stdev_SM

#sorting accuracy in descending order
result_accu=result.sort_values('accu_mean',ascending=False,inplace=False)
#sorting cross entropy in ascending order
result_cross=result.sort_values('cross_mean',ascending=True,inplace=False)

#node 4
#sorting accuracy in descending order
result_accu_4=result.sort_values('accu_mean_4',ascending=False,inplace=False)
#sorting cross entropy in ascending order
result_cross_4=result.sort_values('cross_mean_4',ascending=True,inplace=False)

#node 6
#sorting accuracy in descending order
result_accu_6=result.sort_values('accu_mean_6',ascending=False,inplace=False)
#sorting cross entropy in ascending order
result_cross_6=result.sort_values('cross_mean_6',ascending=True,inplace=False)

#node 9
#sorting accuracy in descending order
result_accu_9=result.sort_values('accu_mean_9',ascending=False,inplace=False)
#sorting cross entropy in ascending order
result_cross_9=result.sort_values('cross_mean_9',ascending=True,inplace=False)

#node SM
#sorting accuracy in descending order
result_accu_SM=result.sort_values('accu_mean_SM',ascending=False,inplace=False)
#sorting cross entropy in ascending order
result_cross_SM=result.sort_values('cross_mean_SM',ascending=True,inplace=False)

#saving outputs
result.to_hdf('optimisation.hd5', key='df', mode='w')
result_accu.to_hdf('optimisation_accu.hd5', key='df', mode='w')
result_cross.to_hdf('optimisation_cross.hd5', key='df', mode='w')
result_accu_4.to_hdf('optimisation_accu_4.hd5', key='df', mode='w')
result_cross_4.to_hdf('optimisation_cross_4.hd5', key='df', mode='w')
result_accu_6.to_hdf('optimisation_accu_6.hd5', key='df', mode='w')
result_cross_6.to_hdf('optimisation_cross_6.hd5', key='df', mode='w')
result_accu_9.to_hdf('optimisation_accu_9.hd5', key='df', mode='w')
result_cross_9.to_hdf('optimisation_cross_9.hd5', key='df', mode='w')
result_accu_SM.to_hdf('optimisation_accu_SM.hd5', key='df', mode='w')
result_cross_SM.to_hdf('optimisation_cross_SM.hd5', key='df', mode='w')
# In[ ]:



