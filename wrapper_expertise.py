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
from gibbsExpertise import gibbs_expertise
from generate_expertise import generate_expertise
from gibbscustom import gibbs_vanilla
nbrun=1
corr_theta_tot_mod=np.zeros([1,nbrun])
corr_phi_tot_mod=np.zeros([1,nbrun])
corr_gamma_tot_mod=np.zeros([1,nbrun])
cosine_theta_tot_mod=np.zeros([1,nbrun])
cosine_phi_tot_mod=np.zeros([1,nbrun])
KL_theta_tot_mod=np.zeros([1,nbrun])
KL_phi_tot_mod=np.zeros([1,nbrun])


corr_theta_tot_cla=np.zeros([1,nbrun])
corr_phi_tot_cla=np.zeros([1,nbrun])
cosine_theta_tot_cla=np.zeros([1,nbrun])
cosine_phi_tot_cla=np.zeros([1,nbrun])
KL_theta_tot_cla=np.zeros([1,nbrun])
KL_phi_tot_cla=np.zeros([1,nbrun])
for i in range(nbrun):
    X,p_generate,theta_generate,phi_generate,zeta,gamma_generate=generate_expertise()
    print("Corpus actuel : ",i)
    corr_theta_mod,cosine_theta_mod,KL_theta_mod,corr_phi_mod,cosine_phi_mod,KL_phi_mod,corr_gamma_mod,v_gamma=gibbs_expertise(X,p_generate,theta_generate,phi_generate,zeta,gamma_generate)
    corr_theta_cla,cosine_theta_cla,KL_theta_cla,corr_phi_cla,cosine_phi_cla,KL_phi_cla=gibbs_vanilla(X,p_generate,theta_generate,phi_generate)
    corr_theta_tot_mod[0,i]=corr_theta_mod
    cosine_theta_tot_mod[0,i]=cosine_theta_mod
    KL_theta_tot_mod[0,i]=KL_theta_mod
    corr_phi_tot_mod[0,i]=corr_phi_mod
    cosine_phi_tot_mod[0,i]=cosine_phi_mod
    KL_phi_tot_mod[0,i]=KL_phi_mod
    corr_gamma_tot_mod[0,i]=corr_gamma_mod
    
    
    corr_theta_tot_cla[0,i]=corr_theta_cla
    cosine_theta_tot_cla[0,i]=cosine_theta_cla
    KL_theta_tot_cla[0,i]=KL_theta_cla
    corr_phi_tot_cla[0,i]=corr_phi_cla
    cosine_phi_tot_cla[0,i]=cosine_phi_cla
    KL_phi_tot_cla[0,i]=KL_phi_cla
    

corr_theta_avg_mod=corr_theta_tot_mod.mean()
corr_phi_avg_mod=corr_phi_tot_mod.mean()
cosine_theta_avg_mod=cosine_theta_tot_mod.mean()
cosine_phi_avg_mod=cosine_phi_tot_mod.mean()
KL_theta_avg_mod=KL_theta_tot_mod.mean()
KL_phi_avg_mod=KL_phi_tot_mod.mean()
corr_gamma_avg_mod=corr_gamma_tot_mod.mean()



corr_theta_avg_cla=corr_theta_tot_cla.mean()
corr_phi_avg_cla=corr_phi_tot_cla.mean()
cosine_theta_avg_cla=cosine_theta_tot_cla.mean()
cosine_phi_avg_cla=cosine_phi_tot_cla.mean()
KL_theta_avg_cla=KL_theta_tot_cla.mean()
KL_phi_avg_cla=KL_phi_tot_cla.mean()



std_corr_theta_mod=np.std(corr_theta_tot_mod)
std_corr_phi_mod=np.std(corr_phi_tot_mod)
std_cosine_theta_mod=np.std(cosine_theta_tot_mod)
std_cosine_phi_mod=np.std(cosine_phi_tot_mod)
std_KL_theta_mod=np.std(KL_theta_tot_mod)
std_KL_phi_mod=np.std(KL_phi_tot_mod)
std_corr_gamma_mod=np.std(corr_gamma_tot_mod)



std_corr_theta_cla=np.std(corr_theta_tot_cla)
std_corr_phi_cla=np.std(corr_phi_tot_cla)
std_cosine_theta_cla=np.std(cosine_theta_tot_cla)
std_cosine_phi_cla=np.std(cosine_phi_tot_cla)
std_KL_theta_cla=np.std(KL_theta_tot_cla)
std_KL_phi_cla=np.std(KL_phi_tot_cla)





