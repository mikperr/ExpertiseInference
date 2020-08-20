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
from lda_train import lda_train
import matplotlib.pyplot as plt
import pandas as pd 

nbrun=1
corr_theta_tot=np.zeros([1,nbrun])
corr_phi_tot=np.zeros([1,nbrun])
cosine_theta_tot=np.zeros([1,nbrun])
cosine_phi_tot=np.zeros([1,nbrun])
KL_theta_tot=np.zeros([1,nbrun])
KL_phi_tot=np.zeros([1,nbrun])
num_topics=3
num_docs=100
term_per_doc=100
voc_size=1000
beta=[0.01 for i in range(voc_size)]
alpha=[1 for i in range(num_topics)]
for i in range(nbrun):
    X,p_generate,theta_generate,phi_generate,data=generate_data(num_topics,num_docs,term_per_doc,voc_size,alpha,beta)
    dct=corpora.Dictionary.load('dct.dict')
    corpus=corpora.MmCorpus('corpus.mm')
    num_words=len(dct)
    print("Corpus actuel : ",i)
    if nbrun==1:
        phi_gensim,corr_theta_gensim,corr_phi_gensim,cosine_theta_gensim,cosine_phi_gensim,KL_theta_gensim,KL_phi_gensim=lda_train(p_generate,theta_generate,phi_generate,num_topics,num_docs)
        phi_cgs,corr_theta,cosine_theta,KL_theta,corr_phi,cosine_phi,KL_phi=gibbs_vanilla(X,p_generate,theta_generate,phi_generate,num_topics,num_docs)
        words_id=np.arange(num_words,dtype=float)
        ymax=max(np.max(phi_generate),np.max(phi_gensim),np.max(phi_cgs))
        fig1=plt.figure()
        for i in range(num_topics):
            plt.subplot(1,num_topics,i+1) 
            plt.bar(words_id,phi_generate[i,:],label="Généré",color="r")
            #plt.subplot(3,num_topics,num_topics+i+1)
            plt.bar(words_id,phi_gensim[i,:],label="Gensim",color="g")
            #plt.subplot(3,num_topics,2*num_topics+i+1)
            plt.bar(words_id,phi_cgs[i,:],label="CGS",color="b")
            plt.legend()
            plt.show()
        fig2=plt.figure()
        for i in range(num_topics):
            plt.subplot(3,num_topics,i+1) 
            plt.bar(words_id,phi_generate[i,:],label="Généré",color="r")
            axes=plt.gca()
            axes.set_ylim([0,ymax])
            plt.legend()
            plt.subplot(3,num_topics,num_topics+i+1)
            plt.bar(words_id,phi_gensim[i,:],label="Gensim",color="g")
            axes=plt.gca()
            axes.set_ylim([0,ymax])
            plt.legend()
            plt.subplot(3,num_topics,2*num_topics+i+1)
            plt.bar(words_id,phi_cgs[i,:],label="CGS",color="b")
            plt.legend()
            plt.show()
            axes=plt.gca()
            axes.set_ylim([0,ymax])
    else:
        corr_theta,cosine_theta,KL_theta,corr_phi,cosine_phi,KL_phi=gibbs_vanilla(X,p_generate,theta_generate,phi_generate,num_topics,num_docs)
        corr_theta_tot[0,i]=corr_theta
        cosine_theta_tot[0,i]=cosine_theta
        KL_theta_tot[0,i]=KL_theta
        corr_phi_tot[0,i]=corr_phi
        cosine_phi_tot[0,i]=cosine_phi
        KL_phi_tot[0,i]=KL_phi

corr_theta_classic=corr_theta_tot.mean()
corr_phi_classic=corr_phi_tot.mean()
cosine_theta_classic=cosine_theta_tot.mean()
cosine_phi_classic=cosine_phi_tot.mean()
KL_theta_classic=KL_theta_tot.mean()
KL_phi_classic=KL_phi_tot.mean()


std_corr_theta_classic=np.std(corr_theta_tot)
std_corr_phi_classic=np.std(corr_phi_tot)
std_cosine_theta_classic=np.std(cosine_theta_tot)
std_cosine_phi_classic=np.std(cosine_phi_tot)
std_KL_theta_classic=np.std(KL_theta_tot)
std_KL_phi_classic=np.std(KL_phi_tot)
