from gensim.models import Word2Vec
import pandas as pd
import numpy as np
spell_check_dict=dict()

def readSpellDict():
	fileName="spell_dict"
	f=open(fileName, 'r')
	for line in f:
		line=line[:-1]
		s=line.split(": ")
		spell_check_dict[s[0]]=s[1]

def calc_w2v_sim(row):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    if row["query"] in spell_check_dict:
    	query=spell_check_dict[row["query"]]
    else:
    	query=row["query"]
    a2 = [x for x in query.lower().split() if x in embedder.vocab]
    b2 = [x for x in row['title'].lower().split() if x in embedder.vocab]
    if len(a2)>0 and len(b2)>0:
        w2v_sim = embedder.n_similarity(a2, b2)
    else:
        return((-1, -1, np.zeros(200)))
    
    vectorA = np.zeros(200)
    for w in a2:
        vectorA += embedder[w]
    vectorA /= len(a2)

    vectorB = np.zeros(200)
    for w in b2:
        vectorB += embedder[w]
    vectorB /= len(b2)

    vector_diff = (vectorA - vectorB)

    w2v_vdiff_dist = np.sqrt(np.sum(vector_diff**2))
    return (w2v_sim, w2v_vdiff_dist, vector_diff)

print "Loading Word2Vec format"
embedder =  Word2Vec.load_word2vec_format("../../data/temp200d.txt",binary=False)

print "Reading csv"
train_df=pd.read_csv("../../data/modified/test_combine.csv")

sim_list = []
dist_list = []
for i,row in train_df.iterrows():
    if i%1000==0:
        print i
    sim, dist, vdiff = calc_w2v_sim(row)
    sim_list.append(sim)
    dist_list.append(dist)
train_df['w2v_sim'] = np.array(sim_list)
train_df['w2v_dist'] = np.array(dist_list)

cols=['title', 'query', 'description','pid']
train_df=train_df.drop(cols,axis=1)
train_df.to_csv("../../data/feat/test_w2v200.csv", index=False, header=True)
