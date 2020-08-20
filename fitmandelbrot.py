# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:15:53 2020

@author: mikap
"""

def fitmandelbrot(phi,num_topics,words_id):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import random
    rank=words_id+1
    #plt.figure()
    #popt_matrix=np.zeros([num_topics,2])
    def mandelbrot(x,a,b):
        return 1/((x+b)**(a))
    #phi_sorted=np.zeros([num_topics,len(phi[0])])
    #for i in range(num_topics):
    #    temp_list=list(phi[i,:])
    #    temp_list.sort(reverse=True)
    #    phi_sorted[i,:]=temp_list
    #    plt.plot(np.log(rank),np.log(phi_sorted[i,:]),label="Sujet %d"%i)
    #    popt, pcov=curve_fit(mandelbrot,rank,phi_sorted[i,:])
    #    popt_matrix[i,:]=popt
    #    plt.plot(np.log(rank),np.log(mandelbrot(rank,*popt)),'--',label="Fit Sujet %d"%i)
    #plt.legend()
    #axes=plt.gca()
    #axes.set_ylim([0,np.log(ymax)])  
    num_folds=100
    perc_train=0.7
    count=0
    rmse=0
    for fold in range(num_folds):
        sample_id=random.sample(list(words_id), int(np.floor(perc_train*len(words_id))))
        sample_id.sort()
        test_id=[i for j, i in enumerate(list(words_id)) if j not in sample_id]
        rank_train=np.array(sample_id)+1
        rank_test=np.array(test_id)+1
        for i in range(num_topics):
            count=count+len(rank_test)
            temp_list=list(phi[i,:])
            temp_list.sort(reverse=True)
            phi_sorted=temp_list
            phi_train=[i for j, i in enumerate(phi_sorted) if j in sample_id]
            phi_test=[i for j, i in enumerate(phi_sorted) if j in test_id]
            popt, pcov=curve_fit(mandelbrot,rank_train,phi_train)
            phi_predict=mandelbrot(rank_test,*popt)
            rmse=rmse+np.sum(np.sqrt((phi_predict-phi_test)**2))
    
    rmse=rmse/count
    erreur=(rmse/np.mean(phi))*100
    #print("RMSE= %.7f"%rmse)
    #print("Erreur= %.2f%%"%erreur)
    return(erreur)