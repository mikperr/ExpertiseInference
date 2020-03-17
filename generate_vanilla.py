# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:47:24 2019

@author: mikap
"""

import math
import random
import numpy
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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
stop_words = stopwords.words('english')
stop_words = stop_words + ['com', 'edu', 'subject', 'lines', 'organization', 'would', 'article', 'could']
## define some constant
TOPIC_N = 3
VOCABULARY_SIZE = 1000
DOC_NUM = 100
TERM_PER_DOC = 100
 
beta = [0.01 for i in range(VOCABULARY_SIZE)]
alpha = [1 for i in range(TOPIC_N)]
FILE_NAME = 'test_expert'

#Creation of my dictionnary
word_list = words.words()
vocab=[]
WL_id=[]
for i in range(VOCABULARY_SIZE):
    draw=random.randint(1,len(word_list))
    while draw in WL_id:
        draw=random.randint(1,len(word_list))
    WL_id.append(draw)
    vocab.append(word_list[draw])

	
   
    
phi = []
theta_matrix=numpy.zeros((DOC_NUM,TOPIC_N))
## generate multinomial distribution over words for each topic
for i in range(TOPIC_N):
	topic =	numpy.random.mtrand.dirichlet(beta, size = 1)
	phi.append(topic)
## generate words for each document
output_f = open(FILE_NAME+'.txt','w')
z_f = open(FILE_NAME+'.z','w')
theta_f = open(FILE_NAME+'.theta','w')
data=[]
#corpus=[]
for i in range(DOC_NUM):
	buffer = {}
	z_buffer = {} ## keep track the true z, which is the topic 
	## first sample theta, the distribution of topics for the given doc
	theta = numpy.random.mtrand.dirichlet(alpha,size = 1)
	theta_matrix[i,]=theta
	for j in range(TERM_PER_DOC):
		## first sample z wrt the distribution of topics in document 
		z = numpy.random.multinomial(1,theta[0],size = 1)
		z_assignment = z.argmax()
		if not z_assignment in z_buffer:
			z_buffer[z_assignment] = 0
		z_buffer[z_assignment] = z_buffer[z_assignment] + 1
		## sample a word from topic z
		w = numpy.random.multinomial(1,phi[z_assignment][0],size = 1)
		w_assignment = w.argmax()
		if not w_assignment in buffer:
			buffer[w_assignment] = 0
		buffer[w_assignment] = buffer[w_assignment] + 1
	data_temp=[] 
	corpus_temp=[] 
    ## output
	output_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
	for word_id, word_count in buffer.items():
		corpus_temp.append((word_id,word_count))
		for h in range(word_count):
			data_temp.append(vocab[word_id])
		output_f.write(str(word_id)+':'+str(word_count)+' ')
	output_f.write('\n')
	data.append(data_temp)
	#corpus.append(corpus_temp)
	z_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
	for z_id, z_count in z_buffer.items():
		z_f.write(str(z_id)+':'+str(z_count)+' ')
	z_f.write('\n')
	theta_f.write(str(i)+'\t')
	for k in range(TOPIC_N):
		theta_f.write(str(k)+':'+str(theta[0][k])+' ')
	theta_f.write('\n')
z_f.close()
theta_f.close()
output_f.close()
 
## output phi
output_f = open(FILE_NAME+'.phi','w')
for i in range(TOPIC_N):
	output_f.write(str(i)+'\t')
	for j in range(VOCABULARY_SIZE):
		output_f.write(str(j)+':'+str(phi[i][0][j])+' ')
	output_f.write('\n')
output_f.close()
 
## output hyper-parameters
output_f = open(FILE_NAME+'.hyper','w')
output_f.write('TOPIC_N:'+str(TOPIC_N)+'\n')
output_f.write('VOCABULARY_SIZE:'+str(VOCABULARY_SIZE)+'\n')
output_f.write('DOC_NUM:'+str(DOC_NUM)+'\n')
output_f.write('TERM_PER_DOC:'+str(TERM_PER_DOC)+'\n')
output_f.write('alpha:'+str(alpha[0])+'\n')
output_f.write('beta:'+str(beta[0])+'\n')
output_f.close()

#Create the dictionnary for the training (other way around)

#for i in range(len(corpus)):
 #   corpus[i]=sorted(corpus[i],key=lambda tup:tup[0])       

#dct = corpora.Dictionary(data)
#dctid=dct.token2id
#corpus = [dct.doc2bow(line) for line in data]

#Save the preprocessing
#dct.save('dct.dict')
#corpora.MmCorpus.serialize('corpus.mm',corpus)
 
 
 # Step 2: Prepare Data (Remove stopwords and lemmatize)
#data_processed = []
#
#for i, doc in enumerate(data):
#    if i==0:
#        print(doc)
#        print(len(doc))
#        
#    print('i=',i)
#    doc_out = []
#    for wd in doc:
#        if wd not in stop_words:  # remove stopwords
#            lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
#            if lemmatized_word:
#                doc_out = doc_out + [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
#        else:
#            continue
#    data_processed.append(doc_out)
# Step 3: Create the Inputs of LDA model: Dictionary and Corpus
dct = corpora.Dictionary(data)
dctid=dct.token2id

corpus = [dct.doc2bow(line) for line in data]

#Save the preprocessing
dct.save('dct.dict')
corpora.MmCorpus.serialize('corpus.mm',corpus)

X_list=[[] for i in range(DOC_NUM)]

for i in range(len(corpus)):
	X_temp=[]
	for j in range(len(corpus[i])):
		for k in range(corpus[i][j][1]):
			X_temp.append(corpus[i][j][0])
	X_list[i]=X_temp
X=numpy.array(X_list)			
numpy.save('X.npy',X)		

#Map wordid from vocab to dct 
wordid=[]
phiid=[[] for i in range(TOPIC_N)]
for i in range(len(dct)):
	for j in range(len(vocab)):
		if dct[i]==vocab[j]:
			wordid.append(j)

for i in range(len(dct)):
	for k in range(len(phi)):
		phiid[k].append(phi[k][0][wordid[i]])

#Find the top N words 		
topN=10
topWords=[[] for i in range(TOPIC_N)]
static=copy.deepcopy(phiid)
for j in range(len(phiid)):
	for i in range(topN):
		m = max(phiid[j])
		idmax=[x for x, y in enumerate(static[j]) if y == m]
		idmaxTemp=[x for x, y in enumerate(phiid[j]) if y == m]
		topWords[j].append(dct[idmax[0]])
		phiid[j].remove(phiid[j][idmaxTemp[0]])
		
#Printing the results 		
for i in range(len(topWords)):
	print("Topic",i)
	print(topWords[i])
	 
phi_matrix=numpy.array(static)
row_sums = phi_matrix.sum(axis=1)
theta_generate=theta_matrix
phi_matrix_new = phi_matrix / row_sums[:, numpy.newaxis]
p_generate=numpy.matmul(theta_matrix,phi_matrix_new)
phi_generate=phi_matrix_new
p_logit_generate=scipy.special.logit(p_generate)
for i in range(p_logit_generate.shape[0]):
	print(i)
	print(p_logit_generate[i,])
numpy.save('p_logit_generate.npy',p_logit_generate)
numpy.save('p_generate.npy',p_generate)
numpy.save('theta_generate.npy',theta_generate)
numpy.save('phi_generate.npy',phi_generate)