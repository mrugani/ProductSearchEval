import pandas as pd

print "Reading files"
cols=['title', 'relevance', 'pid', 'description', 'query', 'values']
df_train_dist=pd.read_csv("../../data/feat/train_distFeat.csv")
df_train_dist=df_train_dist.drop(cols, axis=1)
df_train_counting=pd.read_csv("../../data/feat/train_countingfeat.csv")
# feat=list(df_train_counting.columns)
# for f in feat:
# 	if "pos" in f:
# 		df_train_counting=df_train_counting.drop(f, axis=1)
df_train_counting=df_train_counting.drop(cols, axis=1)
df_train_brand=pd.read_csv("../../data/feat/train_brandFeat.csv")
cols=['query', 'title', 'description', 'values', 'brand', 'relevance', 'pid']
df_train_brand=df_train_brand.drop(cols, axis=1)
df_train_extra = pd.read_csv("../../data/feat/train_extraFeat.csv")
#df_train_merge = pd.read_csv("../../data/feat/train_distfeat_brandFeat_countFeat2.csv")
#df_train_edist = pd.read_csv("../../data/feat/train_extraFeat.csv")
#df_train_edist = df_train_edist.drop(cols, axis=1)
#print df_train_merge.shape
#print df_train_edist.shape
print "Merging files"
df_train_merge=df_train_dist.merge(df_train_counting, how='inner', on='id')
df_train_merge = df_train_merge.merge(df_train_brand, how='inner', on='id')
df_train_merge = df_train_merge.merge(df_train_extra, how='inner', on='id')

#df_train_merge=df_train_merge.merge(df_train_brand, how='left', on='id')
print "Creating csv"
df_train_merge.to_csv("../../data/feat/train_distfeat_brandFeat_countFeat_extraFeat.csv", index=False, header=True)