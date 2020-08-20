# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:55:47 2020

@author: mikap
"""

from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt

quantiles_av=(np.arange(0.01,1,0.01))
quantiles_op=1-quantiles_av
quantiles=np.vstack((quantiles_av,quantiles_op))
alpha1=np.array([0.1, 5])
alpha2=np.array([5, 0.1])
alpha3=np.array([0.1, 0.1])
alpha4=np.array([5,5])
alpha5=np.array([2,4])
alpha6=np.array([1,1])


res1=np.zeros([1, quantiles.shape[1]])
res2=np.zeros([1, quantiles.shape[1]])
res3=np.zeros([1, quantiles.shape[1]])
res4=np.zeros([1, quantiles.shape[1]])
res5=np.zeros([1, quantiles.shape[1]])
res6=np.zeros([1, quantiles.shape[1]])
for i in range(quantiles.shape[1]):
    res1[0][i]=dirichlet.pdf(quantiles[:,i],alpha1)
    res2[0][i]=dirichlet.pdf(quantiles[:,i],alpha2)
    res3[0][i]=dirichlet.pdf(quantiles[:,i],alpha3)
    res4[0][i]=dirichlet.pdf(quantiles[:,i],alpha4)
    res5[0][i]=dirichlet.pdf(quantiles[:,i],alpha5)
    res6[0][i]=dirichlet.pdf(quantiles[:,i],alpha6)
    
res1=res1/np.sum(res1)
res2=res2/np.sum(res2)
res3=res3/np.sum(res3)
res4=res4/np.sum(res4)
res5=res5/np.sum(res5)
res6=res6/np.sum(res6)
plt.plot(quantiles_av,res1[0],label=r'$\alpha=[0.1, 5]$')
plt.plot(quantiles_av,res2[0],label=r'$\alpha=[5, 0.1]$')
plt.plot(quantiles_av,res3[0],label=r'$\alpha=[0.1, 0.1]$')
plt.plot(quantiles_av,res4[0],label=r'$\alpha=[5, 5]$')
plt.plot(quantiles_av,res5[0],label=r'$\alpha=[2, 4]$')
plt.plot(quantiles_av,res6[0],label=r'$\alpha=[1, 1]$')
axes=plt.gca()
axes.set_ylim([0,0.05])
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$P(\theta_1|\alpha)$")
plt.legend()
plt.show
