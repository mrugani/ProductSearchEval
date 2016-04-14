import ngram
import pandas as pd
import re
from utils import try_divide
from  nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

spell_check_dict=dict()

def readSpellDict():
	fileName="spell_dict"
	f=open(fileName, 'r')
	for line in f:
		line=line[:-1]
		s=line.split(": ")
		spell_check_dict[s[0]]=s[1]

def jaccardCoef(A, B):
    #print A, B
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef=try_divide(intersect, union)
    return coef

def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = jaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d

import nltk

stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)

def preprocess_data(line, cat=''):
    line=str(line)
    if line in spell_check_dict:
        line=spell_check_dict[line]
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    if cat=="stem":
        tokens = [ps.stem(x.lower()) for x in token_pattern.findall(line)]
    else:
        tokens = [x.lower() for x in token_pattern.findall(line)]
    result=[]
    for x in tokens:
    	if x not in stopwords:
    		result.append(x);
    return result

#df table of search results
#df table of product descriptions

def extract_distance_features(df):	
    join_str="_"
    df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query"], "stem"), axis=1))
    df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["title"]), axis=1))
    df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["description"]), axis=1))
    df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
    df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
    df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))
    df["attribute_values_unigram"] = list(df.apply(lambda x: preprocess_data(x["values"], "stem"), axis=1))
    df["attribute_values_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["attribute_values_unigram"]), axis=1))
    df["attribute_values_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["attribute_values_unigram"]), axis=1))
    #calculate distance

    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["query", "title", "description", "attribute_values"]
    for dist in dists:
        print "Generating ",dist
        for gram in grams:
            for i in range(len(feat_names)-1):
                for j in range(i+1,len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s"%(dist,gram,target_name,obs_name)] = \
                            list(df.apply(lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1))

    print "Dropping columns"
    df=df.drop(['query_unigram', 'title_unigram', 'description_unigram', 'query_bigram','title_bigram','description_bigram', 'query_trigram', 'title_trigram', 'description_trigram', 'attribute_values_unigram', 'attribute_values_bigram', 'attribute_values_trigram'], axis=1)                      
    print "Creating csv"
    df.to_csv("../../data/feat/test_distFeat.csv", header=True, index=False)
    return df

# df1 = pd.read_csv("../../data/train_pre.csv")
# df2 = pd.read_csv("../../data/attributes_join.csv")
# df=df1.join(df2, on='pid', how='left', rsuffix="pid2")
# #print df.columns
# df=df.drop(['pidpid2'],axis=1)
# df3=pd.read_csv("../../data/product_description_pre.csv")
# df=df.join(df3, on='pid',how='left',rsuffix='pid2')
# df=df.drop(['pidpid2'],axis=1)
# df=extract_distance_features(df)
# df.to_csv("../../data/train_distFeat.csv", header=True, index=False)

readSpellDict()
df1 = pd.read_csv("../../data/modified/test_combine.csv")
extract_distance_features(df1)