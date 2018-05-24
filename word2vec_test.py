
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec


# In[2]:


#model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
#from gensim.models import KeyedVectors
#filename = 'GoogleNews-vectors-negative300.bin'
#model = KeyedVectors.load_word2vec_format(filename, binary=True)



from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'intents/glove.6B.100d.txt'
word2vec_output_file = 'intents/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)



# In[3]:



from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'intents/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
word_vec=model.wv
#word_vec.get_vector('cab')


# In[4]:


x=word_vec.get_vector('pub')
#x.shape


# In[6]:


import numpy as np
cab_file=open('intents/cab.dat','r')
stopwords=open('intents/stopwords.txt','r')


stopwrds=[]
for each in stopwords:
    stopwrds.append(each.strip())
#print(stopwrds)
stopwords.close()
sen=[]
for lines in cab_file:
    sen.append(lines.split())
cleansen=[]
#print(sen)
for sents in sen:
    l=[]
    #print(sents)
    for words in sents:
        if '$cabcompany' in words:
            words='company'
        if words.lower() not in stopwrds and words.lower() in word_vec.vocab:
            l.append(words)
    cleansen.append(l)
#print(cleansen)

#senvec=np.zeros((vec.shape[0],len(sen)))
senvec_cab=[]
#print(senvec.shape)
for words in cleansen:
    vec=np.zeros(word_vec.get_vector('pub').shape)
    for word in words:
        vec=vec+word_vec.get_vector(word)
    senvec_cab.append(vec/len(cleansen))
#print(senvec_cab)
cab_file.close()


# In[9]:


sen_pubs=[]
pubs_file=open('intents/restaurant.dat','r')
for plin in pubs_file:
    sen_pubs.append(plin.split())
    #print(lin)
cleansen_pubs=[]
#print(sen_pubs)
for sents in sen_pubs:
    l=[]
    #print(sents)
    for words in sents:
        if '$' in words:
            words= words[1:]
        if words.lower() not in stopwrds and (words.lower() in word_vec.vocab):
            l.append(words.lower())
    cleansen_pubs.append(l)
#print(cleansen_pubs)

senvec_pubs=[]
#print(senvec.shape)
for words in cleansen_pubs:
    vec=np.zeros(word_vec.get_vector('pub').shape)
    for word in words:
            vec=vec+word_vec.get_vector(word)
    senvec_pubs.append(vec/len(cleansen_pubs))
#print(senvec_pubs)


# In[11]:


sen_res=[]
res_file=open('intents/restart.dat','r')
for eline in res_file:
    sen_res.append(eline.split())
cleansen_res=[]
for i in sen_res:
    l=[]
    for words in i:
        if '$' in words:
            words=words[1:]
        if words.lower() not in stopwrds and (words.lower() in word_vec.vocab):
            l.append(words.lower())
    cleansen_res.append(l)
#print(cleansen_res)
senvec_res=[]
for wrds in cleansen_res:
    vec=np.zeros(word_vec.get_vector('pub').shape)
    for wrd in wrds:
            vec=vec+word_vec.get_vector(wrd)
    senvec_res.append(vec/len(cleansen_res))
#print(senvec_res)


# In[12]:


import scipy
def eqdist(invec,givenvec):
    d=[]
    for every in invec:
        d.append(np.sqrt(sum(np.square(every - givenvec))))
    #print(d)
    return(min(d))
def cosdist(invec,givenvec):
    d=[]
    for every in invec:
        d.append(scipy.spatial.distance.cosine(every,givenvec))
    #print(d)
    return(min(d))


# In[13]:


def getintent(clean_input):
    uinput=clean_input
    givenvec=np.zeros(word_vec.get_vector('pub').shape)
    count=0
    for w in uinput.split():
        if (w.lower() not in stopwrds) and (w.lower() in word_vec.vocab):
            givenvec=givenvec+word_vec.get_vector(w.lower())
            count=count+1
    pubsd=cosdist(senvec_pubs,givenvec)
    cabsd=cosdist(senvec_cab,givenvec)
    red=cosdist(senvec_res,givenvec)
    
    #print(pubsd,cabsd,red)
    if pubsd < cabsd :
        if pubsd < red:
            #print('restaurant')
            return('restaurant')
        else:
            #print('restart')
            return('restart')
    else :
        if cabsd <red:
            #print('Cab')
            return('Cab')
        else:
            #print('restart')
            return('restart')
#print(givenvec/count)

        


# In[14]:


#print("pubs",eqdist(senvec_pubs,givenvec))

#print("cabs",eqdist(senvec_cab,givenvec))


# In[15]:


#getintent('book me a cab')

