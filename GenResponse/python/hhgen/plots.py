import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for plotting predicted histogram, truth histogram error, true/pred plot and calculating scores 
def plotting(y_pred,w_pred,y_truth,w_truth):
    fig=plt.figure(1)
    
    #plotting predicted histogram
    plt.subplot2grid((5,3),(0,0), rowspan=4,colspan=4)
    
    bins=np.arange(0.5,13.5,step=1)
    plt.hist(y_pred,weights=w_pred,bins=bins,alpha=0.5,label='predicted histogram')
    
    #plotting truth histogram error
    htruth=np.histogram(y_truth,weights=w_truth,bins=bins)
    htruth_e=np.histogram(y_truth,weights=w_truth**2,bins=bins)
    err_y=np.sqrt(htruth_e[0])
    x=np.arange(1,13,step=1)
    err_x=np.ones_like(htruth[0])*0.5
    plt.errorbar(x,htruth[0],yerr=err_y,xerr=err_x,ecolor='black',elinewidth='0.5',fmt='none',label='$\sigma_{true}$')
    plt.legend()
    plt.xticks(np.arange(0,14,step=1))
    plt.ylabel('Events')
    
    #plotting truth/pred plot 
    plt.subplot2grid((5,3),(4,0),rowspan=2,colspan=4)
    hpred=np.histogram(y_pred,weights=w_pred,bins=bins)
    ratio=htruth[0]/hpred[0]
    plt.plot(x,ratio,'bo',markersize=2)
    plt.gca().yaxis.grid(True, linestyle='--')       
    plt.ylim(0.8,1.2)
    
    #plotting error in truth/pred plot
    err_y=np.sqrt(htruth_e[0])/hpred[0]
    err_x=np.ones_like(x)*0.5
    plt.errorbar(x,ratio,yerr=err_y,xerr=err_x,ecolor='black',elinewidth='0.5',fmt='none')
    plt.ylabel('true/pred')
    plt.xticks(np.arange(0,14,step=1)) 
    plt.xlabel('Category')
    
    #metrics to test ratio plot
    
    #weighted least squares for truth/pred plot
    rsquare=(ratio-1)**2
    w=1./(err_y)**2
    weighted_least_squares=np.sum(np.multiply(rsquare,w))
    print "Weighted least squares for true/pred plot: ",weighted_least_squares
    
    #spread
    maximum=np.amax(ratio)
    minimum=np.amin(ratio)
    print "Spread for truth/pred plot: ",maximum-minimum
           
    fig.set_size_inches(7, 9)
    plt.show()
    plt.close()