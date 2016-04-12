import ngram
import pandas as pd
import re
from utils import try_divide
from  nltk.stem.porter import PorterStemmer
import nltk
from difflib import SequenceMatcher as seq_matcher
import backports.lzma as lzma

from string import ascii_lowercase

def count(A, B):
    #print A, B
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    return intersect
ps = PorterStemmer()

spell_check_dict=dict()

def readSpellDict():
	fileName="spell_dict"
	f=open(fileName, 'r')
	for line in f:
		line=line[:-1]
		s=line.split(": ")
		spell_check_dict[s[0]]=s[1]

def remove_non_ascii_1(text):
    return ''.join(i for i in text if ord(i)<128)

stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)

def preprocess_data(line):
    line=str(line)
    if line in spell_check_dict:
        line = spell_check_dict[line]
    token_pattern = r"(?u)\b\w\w+\b"	
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    tokens = [ps.stem(x.lower()) for x in token_pattern.findall(line)]
    result=list()
    for x in tokens:
    	if x not in stopwords:
    		result.append(x);
    return result

def last_word(t,q):
	if len(q)==0 or len(t)==0:
		return 0
	if q[-1] in t:
		return 1
	return 0

def compression_distance(x,y,l_x=None,l_y=None):
    if x==y:
        return 0
    x=remove_non_ascii_1(x)
    y=remove_non_ascii_1(y)
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    if l_x is None:
        l_x = len(lzma.compress(x_b))
        l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b+y_b))
    l_yx = len(lzma.compress(y_b+x_b))
    dist = (min(l_xy,l_yx)-min(l_x,l_y))/max(l_x,l_y)
    return dist

def edist(q, t):
	return 1 - seq_matcher(None,q,t).ratio()

def letterFreq(query, title, letter):
	for i in range(26):
		cnt_query=0
		for q in query:
			cnt_query+=q.count(letter)
		cnt_query += 10
		cnt_title=0
		for t in title:
			cnt_title+=t.count(letter)
		cnt_title+=0.5

	return try_divide(try_divide(cnt_query, cnt_title), len(query))

def edist_norm(query, title):
	w=0
	for q in query:
		for t in title:
			lev_dist = seq_matcher(None,q,t).ratio()
			if lev_dist>0.9:
				w+=1

	return try_divide(w, len(query))

def last_word(query, title):
	if len(query)==0 or len(title)==0:
		return 0
	for t in title:
		dist=seq_matcher(None, query[-1], t).ratio()
		if dist > 0.9:
			return 1

	return 0

def extract_brand_features(df):
	print "computing edist"
	#df["edist"]=list(df.apply(lambda x : edist(x["query_original"], x["title"]), axis=1))
	# df["lzma"] = list(df.apply(lambda x: compression_distance(x["query_original"],x["title"]), axis=1))
	df["brand_pre"] = list(df.apply(lambda x: preprocess_data(x["brand"]), axis=1))
	df["count_of_brand"] = list(df.apply(lambda x: len(x["brand_pre"]), axis=1))
	df["attr_unigram"] = list(df.apply(lambda x: preprocess_data(x["attribute_values"]), axis=1))
	df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query_original"]), axis=1))
	df["query_in_brand"] = list(df.apply(lambda x: count(x["brand_pre"], x["query_unigram"]), axis=1))
	df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["title"]), axis=1))
	df["desc_unigram"] = list(df.apply(lambda x: preprocess_data(x["description"]), axis=1))
	df["query_in_title"] = list(df.apply(lambda x: count(x["title_unigram"], x["query_unigram"]), axis=1))
	#df["last_word_in"] = list(df.apply(lambda x: last_word(x["query_unigram"], x["title_unigram"]), axis=1))
	df["query_in_desc"] = list(df.apply(lambda x: count(x["desc_unigram"], x["query_unigram"]), axis=1))
	df["query_in_attr"] = list(df.apply(lambda x: count(x["attr_unigram"], x["query_unigram"]), axis=1))
	df["query_count"] = list(df.apply(lambda x: len(x["query_unigram"]), axis=1))
	df["ratio_title"] = map(try_divide, df["query_in_title"], df["query_count"])
	df["ratio_desc"] = map(try_divide, df["query_in_desc"], df["query_count"])
	#df["query_count_unique"] = list(df.apply(lambda x: len(set(x["query_unigram"])), axis=1))
	df["query_len"] = list(df.apply(lambda x: len(x["query_original"]), axis=1))
	df["title_len"] = list(df.apply(lambda x: len(x["title"]), axis=1))
	#df["edist_norm"] = list(df.apply(lambda x : edist_norm(x["query_unigram"], x["title_unigram"]), axis=1))
	for c in ascii_lowercase:
		df["query_title_"+c]=list(df.apply(lambda x: letterFreq(x["query_unigram"], x["title_unigram"], c), axis=1))

	df=df.drop(['desc_unigram','attr_unigram','title_unigram','query_unigram', 'title', 'query','query_original', 'description','attribute_names','attribute_values', 'brand_pre'], axis=1) 

	df.to_csv("../../data/train_distFeat_updated_10.csv", header=True, index=False)

readSpellDict()
print spell_check_dict
df1 = pd.read_csv("../../data/trial/train_distFeat_updated_1.csv")
#df1 = pd.read_csv("../../data/trial/train_distFeat_counting_updated.csv")
df2 = pd.read_csv("../../data/brands.csv")
df=df1.join(df2, on='pid', how='left', rsuffix="pid2")
extract_brand_features(df)