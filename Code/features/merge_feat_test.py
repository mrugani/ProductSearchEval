import pandas as pd
from utils import try_divide
print "Reading files"
cols=['pid',
 'title',
 'query',
 'description',
 'values']
df_test_dist=pd.read_csv("../../data/feat/test_distFeat.csv")
df_test_dist=df_test_dist.drop(cols, axis=1)
df_test_counting=pd.read_csv("../../data/feat/test_countingFeatFinal.csv")
df_test_counting=df_test_counting.drop('pid_x', axis=1)
df_test_extra=pd.read_csv("../../data/feat/test_extraFeat.csv")
df_test_extra.drop('pid', axis=1)
df_test_brand=pd.read_csv("../../data/feat/test_brandFeat.csv")
df_test_brand = df_test_brand.drop("title", axis=1)
#df_test_edist = df_test_edist.drop(cols, axis=1)
#df_test_merge = pd.read_csv("../../data/feat/test_distfeat_brandFeat_countFeat2.csv")
# df_test_counting1 = pd.read_csv("../../data/feat/test_countingfeat_part2.csv")
# df_test_counting2 = pd.read_csv("../../data/feat/test_countingfeat_part3.csv")
# df_test_counting2.drop("pid", axis=1)
# print "Merging files"
# df_test_counting2=df_test_counting2.merge(df_test_counting1, how='left', on='id')
# df=df_test_counting2
# print "Creating norm features"

# feat_names = ["query", "title", "description"]
# grams = ["unigram", "bigram", "trigram"]

# for gram in grams:
# 		print "Computing for ",gram
# 		for target_name in feat_names:
# 			for obs_name in feat_names:
# 				if target_name != obs_name:
# 					df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
# 					df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
# 					df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
# 					df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
# 					df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])

print "Merging files"

df_test_merge = df_test_dist.merge(df_test_counting, how='inner', on='id')
df_test_merge = df_test_merge.merge(df_test_brand, how='inner', on='id')
df_test_merge = df_test_merge.merge(df_test_extra, how='inner', on='id')



print "Creating csv"
df_test_merge.to_csv("../../data/feat/test_allfeat.csv",header=True, index=False)
print "Columns: ",df_test_merge.columns