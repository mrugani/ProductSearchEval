from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import cPickle
from copy import copy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

def cosine_sim(x, y):
    try:
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print x
        print y
        d = 0.
    return d

def remove_non_ascii_1(text):
    #print text
    return ''.join(i for i in text if ord(i)<128)

def cat_text(x):
    res = '%s %s %s' % (x['query'], x['title'], x['description'])
    res=remove_non_ascii_1(res)
    return res

def tfidfFeature(df):

    column_names = [ "query", "title", "description" ]
    feat_names = [ "query", "title", "description" ]
    feat_names = [ name+"_%s_vocabulary" % ("tfidf") for name in feat_names ]
    # feat_names = [ name+"_%s_vocabulary" % ("tfidf") for name in feat_names ]
    # print "Converting title"
    # df["title"]=list(df["title"].apply(remove_non_ascii_1))
    # print "Converting desc"
    # df["description"]=list(df["description"].apply(remove_non_ascii_1))
    # df["all_text"] = list(df.apply(cat_text, axis=1))
    # tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 3, stop_words = 'english')
    # tf.fit(df["all_text"])
    # vocabulary = tf.vocabulary_
    # feat_names = [ "query", "title", "description" ]
    # feat_names = [ name+"_%s_vocabulary" % ("tfidf") for name in feat_names ]
    # for feat_name,column_name in zip(feat_names, column_names):
    # 	vec=TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 3, stop_words = 'english', vocabulary=vocabulary)
    # 	X_train = vec.fit_transform(df[column_name])
    # 	with open("%s/train.%s.feat.pkl" % (".", feat_name), "wb") as f:
    #         cPickle.dump(X_train, f, -1)
    vec_type="tfidf"
    new_feat_names = copy(feat_names)
    for i in range(len(feat_names)-1):
        for j in range(i+1,len(feat_names)):
            print "generate common %s cosine sim feat for %s and %s" % (vec_type, feat_names[i], feat_names[j])
            for mod in ["train"]:
                with open("%s/%s.%s.feat.pkl" % (".", mod, feat_names[i]), "rb") as f:
                    target_vec = cPickle.load(f)
                with open("%s/%s.%s.feat.pkl" % (".", mod, feat_names[j]), "rb") as f:
                    obs_vec = cPickle.load(f)
                sim = np.asarray(map(cosine_sim, target_vec, obs_vec))[:,np.newaxis]
                ## dump feat
                with open("%s/%s.%s_%s_%s_cosine_sim.feat.pkl" % (".", mod, feat_names[i], feat_names[j], vec_type), "wb") as f:
                    cPickle.dump(sim, f, -1)
            ## update feat names
            new_feat_names.append( "%s_%s_%s_cosine_sim" % (feat_names[i], feat_names[j], vec_type))
    print new_feat_names


df = pd.read_csv("../../data/train_features.csv")
with open("train.query_tfidf_vocabulary_title_tfidf_vocabulary_tfidf_cosine_sim.feat.pkl", "rb") as f:
    vec=cPickle.load(f)
    df["cosine_sim_query_title"]=vec

with open("train.query_tfidf_vocabulary_description_tfidf_vocabulary_tfidf_cosine_sim.feat.pkl", "rb") as f:
    vec=cPickle.load(f)
    df["cosine_sim_query_description"]=vec

with open("train.title_tfidf_vocabulary_description_tfidf_vocabulary_tfidf_cosine_sim.feat.pkl", "rb") as f:
    vec=cPickle.load(f)
    df["cosine_sim_title_description"]=vec
#df=df.drop("all_text")
df.to_csv("../../data/train_features_tf_counting_dist.csv", header=True, index=False)

#tfidfFeature(df)