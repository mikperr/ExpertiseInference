# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:56:57 2019

@author: mikap
"""
def lda_train(p_generate,theta_generate,phi_generate,num_topics,num_docs):
    import matplotlib.pyplot as plt
    from gensim.models import LdaModel, LdaMulticore
    import gensim.downloader as api
    from gensim.utils import simple_preprocess, lemmatize
    import nltk
    from nltk.corpus import stopwords
    from gensim import corpora
    import re
    import pyLDAvis
    import logging
    import numpy as np
    import scipy
    import sys
    from itertools import permutations
    from gensim.models import CoherenceModel
    np.set_printoptions(threshold=sys.maxsize)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    if __name__=='__main__':
        __spec__=None
    #Load dictionnary and corpus
    dct=corpora.Dictionary.load('dct.dict')
    corpus=corpora.MmCorpus('corpus.mm')
    num_words=len(dct)
    # Step 4: Train the LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=None,
                         num_topics=num_topics, 
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True,
                         minimum_probability=0)
    
    # save the model
    lda_model.save('lda_model.model')
    
    # See the topics
    i=0
    theta_matrix=np.zeros((num_docs,num_topics))
    for c in lda_model[corpus]:
        print(i)
        print("Document Topics      : ", c[0])      # [(Topics, Perc Contrib)]
        for j in range(theta_matrix.shape[1]):
    	    theta_matrix[i,j]=c[0][j][1]
        i=i+1
    	
    #    print("Word id, Topics      : ", c[1][:])  # [(Word id, [Topics])]
        #print("Phi Values (word id) : ", c[2][:])  # [(Word id, [(Topic, Phi Value)])]
    #    print("Word, Topics         : ", [(dct[wd], topic) for wd, topic in c[1][:]])   # [(Word, [Topics])]
    #    print("Phi Values (word)    : ", [(dct[wd], topic) for wd, topic in c[2][:]])  # [(Word, [(Topic, Phi Value)])]
    #    print("------------------------------------------------------\n")
    
    
    for j in range(num_topics):
    	print("Topic", j)
    	for i in range(len(lda_model.get_topic_terms(j, 10))):
    		print(dct[lda_model.get_topic_terms(j, 10)[i][0]],lda_model.get_topic_terms(j, 10)[i][1])
    
    phi_matrix=lda_model.get_topics()
    row_sums = theta_matrix.sum(axis=1)
    theta_matrix_new = theta_matrix / row_sums[:, np.newaxis]
    p=np.matmul(theta_matrix_new,phi_matrix)
    p_logit=scipy.special.logit(p)
    
    for i in range(p_logit.shape[0]):
    	print(i)
    	print(p_logit[i,])
    p_logit_generate=np.load('p_logit_generate.npy')
    p_generate=np.load('p_generate.npy')
    theta_generate=np.load('theta_generate.npy')
    phi_generate=np.load('phi_generate.npy')
    corr_p=np.zeros((1,num_docs))
    corr_p_logit=np.zeros((1,num_docs))
    cosine_p=np.zeros((1,num_docs))
    for i in range(p_logit.shape[0]):
    	corr_p_logit[0,i]=np.corrcoef(p_logit[i,],p_logit_generate[i,])[1,0]
    	corr_p[0,i]=np.corrcoef(p[i,],p_generate[i,])[1,0]
    	cosine_p[0,i]=scipy.spatial.distance.cosine(p[i,],p_generate[i,])
    corr_avg_p_inter=np.mean(corr_p)
    cosine_avg_p_inter=np.mean(cosine_p)
    corr_avg_p_logit_inter=np.mean(corr_p_logit)
    corr_avg_p_wordDist=np.mean(np.corrcoef(p)) #Average of the correlation matrix for the word distributions of each documents (shape numDocxnumDoc)
    corr_avg_p_docDist=np.mean(np.corrcoef(np.transpose(p))) #Average of the correlation matrix for the document distributions for each words (shape dictLenxdictLen) 
      
    corr_avg_pgenerate_wordDist=np.mean(np.corrcoef(p_generate)) #Average of the correlation matrix for the word distributions of each documents (shape numDocxnumDoc)
    corr_avg_pgenerate_docDist=np.mean(np.corrcoef(np.transpose(p_generate))) #Average of the correlation matrix for the document distributions for each words (shape dictLenxdictLen) 
      
    
    theta=theta_matrix_new
    phi=phi_matrix
    
    #This section is to compile to correlation and cosine of each column arrangment combination of a 3 topic model (theta_matrix)			
    compilation_corr_theta=[]
    compilation_cosine_theta=[]
    compilation_corr_phi=[]
    compilation_cosine_phi=[]
    compilation_KL_theta=[]
    compilation_KL_phi=[]
    
    l = list(permutations(range(1, num_topics+1)))
    
    for combi in range(len(l)):
        v_theta=np.zeros([num_docs,num_topics])
        v_phi=np.zeros([num_topics,num_words])
        for tid in range(num_topics):
            v_theta[:,tid]=theta[:,l[combi][tid]-1]
            v_phi[tid,:]=phi[l[combi][tid]-1,:]
        corr_theta=np.zeros((1,num_docs))
        cosine_theta=np.zeros((1,num_docs))
        KL_theta=np.zeros((1,num_docs))
        corr_phi=np.zeros((1,num_topics))
        cosine_phi=np.zeros((1,num_topics))
        KL_phi=np.zeros((1,num_topics))
        
        for i in range(theta_generate.shape[0]):
            corr_theta[0,i]=np.corrcoef(v_theta[i,:],theta_generate[i,:])[1,0]
            cosine_theta[0,i]=scipy.spatial.distance.cosine(v_theta[i,:],theta_generate[i,:])
            KL_theta[0,i]=scipy.stats.entropy(theta_generate[i,:],v_theta[i,:])
        compilation_corr_theta.append(corr_theta.mean())
        compilation_cosine_theta.append(cosine_theta.mean())
        compilation_KL_theta.append(KL_theta.mean())
        for i in range(phi_generate.shape[0]):
            corr_phi[0,i]=np.corrcoef(v_phi[i,:],phi_generate[i,:])[1,0]
            cosine_phi[0,i]=scipy.spatial.distance.cosine(v_phi[i,:],phi_generate[i,:])
            KL_phi[0,i]=scipy.stats.entropy(phi_generate[i,:],v_phi[i,:])
        compilation_corr_phi.append(corr_phi.mean())
        compilation_cosine_phi.append(cosine_phi.mean())
        compilation_KL_phi.append(KL_phi.mean())
            
    compilation_cosine_phi=np.array(compilation_cosine_phi)
    compilation_corr_phi=np.array(compilation_corr_phi)
    compilation_KL_phi=np.array(compilation_KL_phi)
    compilation_cosine_theta=np.array(compilation_cosine_theta)
    compilation_corr_theta=np.array(compilation_corr_theta)
    compilation_KL_theta=np.array(compilation_KL_theta)
    
    alignment=compilation_KL_phi.argmin()
    if alignment != compilation_cosine_phi.argmin() | alignment != compilation_cosine_theta.argmin() | alignment != compilation_corr_theta.argmax() | alignment != compilation_corr_phi.argmax() | alignment != compilation_KL_theta.argmin():
        print('Warning!!! The alignments are not coherents.')
    
    #Determining the final correlation and cosine values 
    v_theta=np.zeros([num_docs,num_topics])
    v_phi=np.zeros([num_topics,num_words])
    for tid in range(num_topics):
        v_theta[:,tid]=theta[:,l[alignment][tid]-1]
        v_phi[tid,:]=phi[l[alignment][tid]-1,:]
    corr_theta=np.zeros((1,num_docs))
    cosine_theta=np.zeros((1,num_docs))
    KL_theta=np.zeros((1,num_docs))
    corr_phi=np.zeros((1,num_topics))
    cosine_phi=np.zeros((1,num_topics))
    KL_phi=np.zeros((1,num_topics))
    for i in range(theta_generate.shape[0]):
        corr_theta[0,i]=np.corrcoef(v_theta[i,:],theta_generate[i,:])[1,0]
        cosine_theta[0,i]=scipy.spatial.distance.cosine(v_theta[i,:],theta_generate[i,:])
        KL_theta=scipy.stats.entropy(theta_generate[i,:],v_theta[i,:])
    for i in range(phi_generate.shape[0]):
        corr_phi[0,i]=np.corrcoef(v_phi[i,:],phi_generate[i,:])[1,0]
        cosine_phi[0,i]=scipy.spatial.distance.cosine(v_phi[i,:],phi_generate[i,:])
        KL_phi=scipy.stats.entropy(phi_generate[i,:],v_phi[i,:])
    corr_theta=corr_theta.mean()
    cosine_theta=cosine_theta.mean()
    KL_theta=KL_theta.mean()
    corr_phi=corr_phi.mean()
    cosine_phi=cosine_phi.mean()
    KL_phi=KL_phi.mean()
    words_id=np.arange(num_words)
    #coherence_model_lda=CoherenceModel(model=lda_model,texts=corpus,dictionary=dct,coherence='c_v')
    #coherence_lda=coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)
    return(v_phi,corr_theta,corr_phi,cosine_theta,cosine_phi,KL_theta,KL_phi)