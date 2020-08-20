# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:22:00 2020

@author: mikap
"""
def gibbs_expertise(X,p_generate,theta_generate,phi_generate,zeta_generate,gamma_generate):
    from itertools import permutations
    import numpy as np
    import random
    from gensim import corpora
    import scipy
    import matplotlib.pyplot as plt
    dct=corpora.Dictionary.load('dct.dict');
    def sigmoid(x,a,b):
      return (1 / (1 + np.exp(-a*(x-b))))
    a=50
#    np.load('X_mat.npy');
#    np.load('p_generate.npy')
#    np.load('theta_generate.npy')
#    np.load('phi_generate.npy')
#    np.load('zeta_generate.npy')
#    np.load('gamma_generate.npy')
    num_topics=3;
    num_words=len(dct);
    
    beta=1;
    alpha=1;
    DOC_PER_PERS=20
    num_pers=5
    num_docs=DOC_PER_PERS*num_pers
    
    #Initialisation aléatoire
    C_WT=np.zeros([num_words,num_topics],dtype=int);
    C_DT=np.zeros([num_docs,num_topics],dtype=int);
    topic_assignment=np.zeros([X.shape[0],X.shape[1]],dtype=int);
    
    for d in range(X.shape[0]):
        for w in range(X.shape[1]):
            t=random.randint(0,num_topics-1)
            topic_assignment[d,w]=t;
            C_WT[X[d,w],t]=C_WT[X[d,w],t]+1
            C_DT[d,t]=C_DT[d,t]+1
    
    #Gibbs Sampling 
    it=50;
    gamma=zeta_generate.mean()*np.ones([num_pers,num_topics])	
    
                
    
    
    for i in range(it):
        completion=int(np.floor((i/it)*100))
        print(completion,"%")
    
        for d in range(X.shape[0]):
            pers_id=np.int(np.floor(d/DOC_PER_PERS))
            for w in range(X.shape[1]):
                
    		
                C_WT[X[d,w],topic_assignment[d,w]]=C_WT[X[d,w],topic_assignment[d,w]]-1
                C_DT[d,topic_assignment[d,w]]=C_DT[d,topic_assignment[d,w]]-1
                proba=[]
                for j in range(num_topics):
                    left=(C_WT[X[d,w],j]+beta)/(C_WT[:,j].sum()+num_words*beta);
                    right=(C_DT[d,j]+alpha)/(C_DT[d,:].sum()+num_topics*alpha);
                    #smooth=1/sigmoid(zeta_generate[0,X[d,w]],a,gamma[pers_id,j])
                    proba.append(left*right)
                    proba_norm=[]
                for h in range(len(proba)):
                    proba_norm.append(proba[h]/sum(proba))
                    t_sample=np.random.multinomial(1,proba_norm,size = 1)
                new_assignment=t_sample.argmax()
                C_WT[X[d,w],new_assignment]=C_WT[X[d,w],new_assignment]+1
                C_DT[d,new_assignment]=C_DT[d,new_assignment]+1
                topic_assignment[d,w]=new_assignment
    print("100%")
    for d in range(X.shape[0]):
        pers_id=np.int(np.floor(d/DOC_PER_PERS))
        for j in range(num_topics):
            #rc=topic_assignment[d,:]==j
            rc=21
            if rc>20:
                #zeta_min=np.float64(np.arange(5,10))
                zeta_min=6
                for w in range(X.shape[1]):
                    if topic_assignment[d,w]==j:
                        
                        #arg=np.argwhere(zeta_min==max(zeta_min))[0][0]
                        #zeta_min[arg]=(min(zeta_min[arg],zeta_generate[0,X[d,w]]))
                        zeta_min=(min(zeta_min,zeta_generate[0,X[d,w]]))
                        
                #if gamma[pers_id,j]==0:
                #    gamma[pers_id,j]=np.mean(zeta_min)
                #else:
                #    gamma[pers_id,j]=np.mean([gamma[pers_id,j],np.mean(zeta_min)])
                gamma[pers_id,j]=zeta_min
    
    
    #for d in range(X.shape[0]):
    #    pers_id=np.int(np.floor(d/DOC_PER_PERS))
    #    for j in range(num_topics):
    #        rc=topic_assignment[d,:]==j
    #        if rc.sum()>20:
    #            zeta_min=np.float64(np.arange(5,10))
    #            for w in range(X.shape[1]):
    #                if topic_assignment[d,w]==j:
    #                    arg=np.argwhere(zeta_min==max(zeta_min))[0][0]
    #                    zeta_min[arg]=(min(zeta_min[arg],zeta_generate[X[d,w]]))
    #            if gamma[pers_id,j]==0:
    #                gamma[pers_id,j]=np.mean(zeta_min)
    #            else:
    #                gamma[pers_id,j]=np.mean([gamma[pers_id,j],np.mean(zeta_min)])
    phi=np.zeros([num_words,num_topics])
    theta=np.zeros([num_docs,num_topics])
    for j in range(phi.shape[1]):
        row_sum=C_WT[:,j].sum()
        for i in range(phi.shape[0]):
    		      phi[i,j]=(C_WT[i,j]+beta)/(row_sum+num_words*beta)
    phi=np.transpose(phi)		
    for i in range(theta.shape[0]):
    	   col_sum=C_DT[i,:].sum()
    	   for j in range(theta.shape[1]):
    		      theta[i,j]=(C_DT[i,j]+alpha)/(col_sum+num_topics*alpha)
    				
    p=np.matmul(theta,phi)
    corr_p=np.zeros((1,num_docs))
    corr_p_logit=np.zeros((1,num_docs))
    p_logit=scipy.special.logit(p)
    p_logit_generate=scipy.special.logit(p_generate)
    for i in range(p.shape[0]):
        corr_p[0,i]=np.corrcoef(p[i,],p_generate[i,])[1,0]
        corr_p_logit[0,i]=np.corrcoef(p_logit[i,],p_logit_generate[i,])[1,0]
        cosine_p=scipy.spatial.distance.cosine(p[i,],p_generate[i,])
    corr_avg_p_inter=np.mean(corr_p)
    corr_avg_p_wordDist=np.mean(np.corrcoef(p)) #Average of the correlation matrix for the word distributions of each documents (shape numDocxnumDoc)
    corr_avg_p_docDist=np.mean(np.corrcoef(np.transpose(p))) #Average of the correlation matrix for the document distributions for each words (shape dictLenxdictLen) 
    
    corr_avg_p_logit_inter=np.mean(corr_p_logit)
    cosine_avg_p_inter=np.mean(cosine_p)
    corr_avg_pgenerate_wordDist=np.mean(np.corrcoef(p_generate)) #Average of the correlation matrix for the word distributions of each documents (shape numDocxnumDoc)
    corr_avg_pgenerate_docDist=np.mean(np.corrcoef(np.transpose(p_generate))) #Average of the correlation matrix for the document distributions for each words (shape dictLenxdictLen) 		
    
     	
    		
    				
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
    v_gamma=np.zeros([num_pers,num_topics])
    for tid in range(num_topics):
        v_theta[:,tid]=theta[:,l[alignment][tid]-1]
        v_phi[tid,:]=phi[l[alignment][tid]-1,:]
        v_gamma[:,tid]=gamma[:,l[alignment][tid]-1]
    corr_theta=np.zeros((1,num_docs))
    corr_gamma=np.zeros((1,num_docs))
    cosine_theta=np.zeros((1,num_docs))
    KL_theta=np.zeros((1,num_docs))
    corr_phi=np.zeros((1,num_topics))
    cosine_phi=np.zeros((1,num_topics))
    KL_phi=np.zeros((1,num_topics))
    for i in range(theta_generate.shape[0]):
        corr_theta[0,i]=np.corrcoef(v_theta[i,:],theta_generate[i,:])[1,0]
        cosine_theta[0,i]=scipy.spatial.distance.cosine(v_theta[i,:],theta_generate[i,:])
        KL_theta=scipy.stats.entropy(theta_generate[i,:],v_theta[i,:])
    for i in range(gamma_generate.shape[0]):
        corr_gamma[0,i]=np.corrcoef(v_gamma[i,:],gamma_generate[i,:])[1,0]
    for i in range(phi_generate.shape[0]):
        corr_phi[0,i]=np.corrcoef(v_phi[i,:],phi_generate[i,:])[1,0]
        cosine_phi[0,i]=scipy.spatial.distance.cosine(v_phi[i,:],phi_generate[i,:])
        KL_phi=scipy.stats.entropy(phi_generate[i,:],v_phi[i,:])
    corr_theta=corr_theta.mean()
    cosine_theta=cosine_theta.mean()
    KL_theta=KL_theta.mean()
    corr_gamma=corr_gamma.mean()
     
    corr_phi=corr_phi.mean()
    cosine_phi=cosine_phi.mean()
    KL_phi=KL_phi.mean()
    words_id=np.arange(num_words)
#    plt.clf()
#    for i in range(num_topics):
#        plt.subplot(2,num_topics,i+1)
#        plt.bar(words_id,phi_generate[i,:])
#        plt.subplot(2,num_topics,num_topics+i+1)
#        plt.bar(words_id,v_phi[i,:])
#    plt.show()
    return corr_theta,cosine_theta,KL_theta,corr_phi,cosine_phi,KL_phi,corr_gamma,v_gamma