from sklearn import model_selection

def optimize_parameters_randomizedCV(classifier,X_train,y_train,param_grid,cvOpt=3,weights=None,nJobs=10,nIter=10):
    
    #evaluating score of classifier by cross validation
    scores=model_selection.cross_val_score(classifier,X_train, y_train,scoring="roc_auc",n_jobs=nJobs,cv=cvOpt)                    
    print "-Initial Accuracy-"
    print "Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std()) 
    
    #randomised search on hyper parameters
    if weights==None:
        clf=model_selection.RandomizedSearchCV(classifier,param_grid,n_iter=nIter,cv=cvOpt,scoring='roc_auc',n_jobs=nJobs,
                                                 verbose=1)
    else:
        clf=model_selection.RandomizedSearchCV(classifier,param_grid,n_iter=nIter,cv=cvOpt,scoring='roc_auc',n_jobs=nJobs,
                                                 verbose=1,fit_params={'sample_weight': weights})
                              
    clf.fit(X_train, y_train)
    
    print "Best parameter set found on development set:"
    print clf.best_estimator_
    print "Grid scores on a subset of the development set:"
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.4f (+/-%0.04f) for %r"%(mean_score, scores.std(), params)
    
    return clf.grid_scores_