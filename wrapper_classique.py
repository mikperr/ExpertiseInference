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
from gibbscustom import gibbs_vanilla
from generate_data import generate_data
nbrun=100
corr_theta_tot=np.zeros([1,nbrun])
corr_phi_tot=np.zeros([1,nbrun])
cosine_theta_tot=np.zeros([1,nbrun])
cosine_phi_tot=np.zeros([1,nbrun])
KL_theta_tot=np.zeros([1,nbrun])
KL_phi_tot=np.zeros([1,nbrun])
for i in range(nbrun):
    X,p_generate,theta_generate,phi_generate=generate_data()
    print("Corpus actuel : ",i)
    corr_theta,cosine_theta,KL_theta,corr_phi,cosine_phi,KL_phi=gibbs_vanilla(X,p_generate,theta_generate,phi_generate)
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
KL_phi_classic=KL_theta_tot.mean()


std_corr_theta_classic=np.std(corr_theta_tot)
std_corr_phi_classic=np.std(corr_phi_tot)
std_cosine_theta_classic=np.std(cosine_theta_tot)
std_cosine_phi_classic=np.std(cosine_phi_tot)
std_KL_theta_classic=np.std(KL_theta_tot)
std_KL_phi_classic=np.std(KL_theta_tot)
