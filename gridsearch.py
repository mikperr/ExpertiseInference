# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:53:46 2020

@author: mikap
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:09:03 2020

@author: mikap
"""
import math
import random
import numpy as np
import numpy.random
import sys
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import re
import logging
from nltk.corpus import words
import random
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import re
import logging
import copy
import scipy
from gibbsvanilla import gibbs_vanilla
from generate_data import generate_data
from fitzipf import fitzipf
from fitmandelbrot import fitmandelbrot
from lda_train import lda_train
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

nbrun=10
corr_theta_tot=np.zeros([1,nbrun])
corr_phi_tot=np.zeros([1,nbrun])
cosine_theta_tot=np.zeros([1,nbrun])
cosine_phi_tot=np.zeros([1,nbrun])
KL_theta_tot=np.zeros([1,nbrun])
KL_phi_tot=np.zeros([1,nbrun])
KL_phi_tot_gensim=np.zeros([1,nbrun])
KL_theta_tot_gensim=np.zeros([1,nbrun])
erreur_tot_zipf=np.zeros([1,nbrun])
erreur_tot_mand=np.zeros([1,nbrun])
num_topics=3
num_docs=100
term_per_doc=100
voc_size=1000
#betas=[0.01, 0.03, 0.05, 0.07,1,1.5,2,3,5]
betas=[0.01, 0.1, 0.5, 0.7,1,1.5,2,3,5]
#betas=[0.01,0.03]
alphas=betas
KL_theta_cgs=np.zeros([1,len(betas)*len(alphas)])
KL_phi_cgs=np.zeros([1,len(betas)*len(alphas)])
KL_theta_postgensim=np.zeros([1,len(betas)*len(alphas)])
KL_phi_postgensim=np.zeros([1,len(betas)*len(alphas)])
erreur_post_zipf=np.zeros([1,len(betas)*len(alphas)])
erreur_post_mand=np.zeros([1,len(betas)*len(alphas)])
count=-1
beta_heatmap=np.zeros([1,len(betas)*len(alphas)])
alpha_heatmap=np.zeros([1,len(betas)*len(alphas)])
for k in range(len(betas)):
    for j in range(len(alphas)):
        alpha=[alphas[j] for i in range(num_topics)]
        beta=[betas[k] for i in range(voc_size)]
        count=count+1
        alpha_heatmap[0,count]=alpha[0]
        beta_heatmap[0,count]=beta[0]
        for i in range(nbrun):
            print("1")
            X,p_generate,theta_generate,phi_generate,data=generate_data(num_topics,num_docs,term_per_doc,voc_size,alpha,beta)
            dct=corpora.Dictionary.load('dct.dict')
            corpus=corpora.MmCorpus('corpus.mm')
            num_words=len(dct)
            print("2")
            phi_gensim,corr_theta_gensim,corr_phi_gensim,cosine_theta_gensim,cosine_phi_gensim,KL_theta_gensim,KL_phi_gensim=lda_train(p_generate,theta_generate,phi_generate,num_topics,num_docs)
            words_id=np.arange(num_words,dtype=float)
            print("3")
            erreur_tot_zipf[0,i]=fitzipf(phi_generate,num_topics,words_id)
            erreur_tot_mand[0,i]=fitmandelbrot(phi_generate,num_topics,words_id)
            print("4")
            print("Combi actuelle : ",count)
            print("Run actuelle : ",i)
            print("5")
            phi_cgs,corr_theta,cosine_theta,KL_theta,corr_phi,cosine_phi,KL_phi=gibbs_vanilla(X,p_generate,theta_generate,phi_generate,num_topics,num_docs)
            print("6")
            corr_theta_tot[0,i]=corr_theta
            cosine_theta_tot[0,i]=cosine_theta
            KL_theta_tot[0,i]=KL_theta
            KL_theta_tot_gensim[0,i]=KL_theta_gensim
            corr_phi_tot[0,i]=corr_phi
            cosine_phi_tot[0,i]=cosine_phi
            KL_phi_tot[0,i]=KL_phi
            KL_phi_tot_gensim[0,i]=KL_phi_gensim
        
        corr_theta_cgs=corr_theta_tot.mean()
        corr_phi_cgs=corr_phi_tot.mean()
        cosine_theta_cgs=cosine_theta_tot.mean()
        cosine_phi_cgs=cosine_phi_tot.mean()
        KL_theta_cgs[0,count]=KL_theta_tot.mean()
        KL_phi_cgs[0,count]=KL_phi_tot.mean()
        KL_theta_postgensim[0,count]=KL_theta_tot_gensim.mean()
        KL_phi_postgensim[0,count]=KL_phi_gensim.mean()
        erreur_post_zipf[0,count]=erreur_tot_zipf.mean()
        erreur_post_mand[0,count]=erreur_tot_mand.mean()
        
        
        std_corr_theta_cgs=np.std(corr_theta_tot)
        std_corr_phi_cgs=np.std(corr_phi_tot)
        std_cosine_theta_cgs=np.std(cosine_theta_tot)
        std_cosine_phi_cgs=np.std(cosine_phi_tot)
        std_KL_theta_cgs=np.std(KL_theta_tot)
        std_KL_phi_cgs=np.std(KL_phi_tot)
beta_heatmap_df=beta_heatmap.tolist()[-1]
alpha_heatmap_df=alpha_heatmap.tolist()[-1]
KL_theta_postgensim_df=KL_theta_postgensim.tolist()[-1]
KL_phi_postgensim_df=KL_phi_postgensim.tolist()[-1]
KL_phi_cgs_df=KL_phi_cgs.tolist()[-1]
KL_theta_cgs_df=KL_theta_cgs.tolist()[-1]
erreur_zipf_df=erreur_post_zipf.tolist()[-1]
erreur_mand_df=erreur_post_mand.tolist()[-1]
df_theta_gensim=pd.DataFrame(list(zip(beta_heatmap_df,alpha_heatmap_df,KL_theta_postgensim_df)),columns=["beta", "alpha","KL"])
df_phi_gensim=pd.DataFrame(list(zip(beta_heatmap_df,alpha_heatmap_df,KL_phi_postgensim_df)),columns=["beta", "alpha","KL"])
df_phi_cgs=pd.DataFrame(list(zip(beta_heatmap_df,alpha_heatmap_df,KL_phi_cgs_df)),columns=["beta", "alpha","KL"])
df_theta_cgs=pd.DataFrame(list(zip(beta_heatmap_df,alpha_heatmap_df,KL_theta_cgs_df)),columns=["beta", "alpha","KL"])
df_erreur_zipf=pd.DataFrame(list(zip(beta_heatmap_df,alpha_heatmap_df,erreur_zipf_df)),columns=["beta", "alpha","Erreur (%)"])
df_erreur_mand=pd.DataFrame(list(zip(beta_heatmap_df,alpha_heatmap_df,erreur_mand_df)),columns=["beta", "alpha","Erreur (%)"])
df_theta_gensim_hm=df_theta_gensim.pivot("beta","alpha","KL")
df_phi_gensim_hm=df_phi_gensim.pivot("beta","alpha","KL")
df_theta_cgs_hm=df_theta_cgs.pivot("beta","alpha","KL")
df_phi_cgs_hm=df_phi_cgs.pivot("beta","alpha","KL")
df_erreur_zipf_hm=df_erreur_zipf.pivot("beta","alpha","Erreur (%)")
df_erreur_mand_hm=df_erreur_mand.pivot("beta","alpha","Erreur (%)")
plt.figure()
sns.heatmap(df_theta_gensim_hm,cbar_kws={'label':'KL'})
plt.figure()
sns.heatmap(df_phi_gensim_hm,cbar_kws={'label':'KL'})
plt.figure()
sns.heatmap(df_theta_cgs_hm,cbar_kws={'label':'KL'})
plt.figure()
sns.heatmap(df_phi_cgs_hm,cbar_kws={'label':'KL'})
plt.figure()
sns.heatmap(df_erreur_zipf_hm,cbar_kws={'label':'Erreur (%)'})
plt.figure()
sns.heatmap(df_erreur_mand_hm,cbar_kws={'label':'Erreur (%)'})

f,(ax1,ax2,axcb)=plt.subplots(1,3,gridspec_kw={'width_ratios':[1,1,0.08]})
ax1.title.set_text('Gensim')
ax2.title.set_text('CGS')
ax1.get_shared_y_axes().join(ax2)
g1=sns.heatmap(df_theta_gensim_hm,cbar=False,ax=ax1,vmin=0,vmax=max(df_theta_gensim['KL'].max(),df_theta_cgs['KL'].max()))
g2=sns.heatmap(df_theta_cgs_hm,cbar_kws={'label':'KL'},ax=ax2,cbar_ax=axcb,vmin=0,vmax=max(df_theta_gensim['KL'].max(),df_theta_cgs['KL'].max()))


f,(ax1,ax2,axcb)=plt.subplots(1,3,gridspec_kw={'width_ratios':[1,1,0.08]})
ax1.title.set_text('Gensim')
ax2.title.set_text('CGS')
ax1.get_shared_y_axes().join(ax2)
g1=sns.heatmap(df_phi_gensim_hm,cbar=False,ax=ax1,vmin=0,vmax=max(df_phi_gensim['KL'].max(),df_phi_cgs['KL'].max()))
g2=sns.heatmap(df_phi_cgs_hm,cbar_kws={'label':'KL'},ax=ax2,cbar_ax=axcb,vmin=0,vmax=max(df_phi_gensim['KL'].max(),df_phi_cgs['KL'].max()))

f,(ax1,ax2,axcb)=plt.subplots(1,3,gridspec_kw={'width_ratios':[1,1,0.08]})
ax1.title.set_text('Zipf')
ax2.title.set_text('Mandelbrot')
ax1.get_shared_y_axes().join(ax2)
g1=sns.heatmap(df_erreur_zipf_hm,cbar=False,ax=ax1,vmin=0,vmax=max(df_erreur_zipf['Erreur (%)'].max(),df_erreur_mand['Erreur (%)'].max()))
g2=sns.heatmap(df_erreur_mand_hm,cbar_kws={'label':'Erreur (%)'},ax=ax2,cbar_ax=axcb,vmin=0,vmax=max(df_erreur_zipf['Erreur (%)'].max(),df_erreur_mand['Erreur (%)'].max()))
