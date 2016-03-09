import ngram
import pandas as pd
import re
from utils import try_divide

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


def preprocess_data(line):
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    return tokens

#df table of search results
#df table of product descriptions
def extract_distance_features(df):	
    join_str="_"
    df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query"]), axis=1))
    df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["title"]), axis=1))
    df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["description"]), axis=1))
    df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
    df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
    df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

    #calculate distance

    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["query", "title", "description"]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names)-1):
                for j in range(i+1,len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s"%(dist,gram,target_name,obs_name)] = \
                            list(df.apply(lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1))

    df=df.drop(['query_unigram', 'title_unigram', 'description_unigram', 'query_bigram','title_bigram','description_bigram', 'query_trigram', 'title_trigram', 'description_trigram'], axis=1)                      
    return df

df1 = pd.read_csv("../../data/train_pre.csv")
df2 = pd.read_csv("../../data/product_description_pre.csv")
df=pd.merge(left=df1, right=df2, how='left', left_on="pid", right_on="pid")
#print df.columns
df=extract_distance_features(df)
df.to_csv("../../data/train_join.csv", header=True, index=False)
