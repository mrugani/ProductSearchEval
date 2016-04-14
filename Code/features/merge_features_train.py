import pandas as pd

print "Reading files"
cols=['query', 'title', 'description', 'values', 'relevance']
df_train_dist=pd.read_csv("../../data/feat/train_distFeat.csv")
df_train_dist=df_train_dist.drop(cols, axis=1)
df_train_counting=pd.read_csv("../../data/feat/train_countingfeat.csv")
feat=list(df_train_counting.columns)
for f in feat:
	if "div" in f or "pos" in f:
		df_train_counting=df_train_counting.drop(f, axis=1)
df_train_counting=df_train_counting.drop(cols, axis=1)
df_train_brand=pd.read_csv("../../data/feat/train_brandFeat.csv")
cols=['query', 'title', 'description', 'values', 'brand']
df_train_brand=df_train_brand.drop(cols, axis=1)
print "Merging files"
df_train_merge=df_train_counting.merge(df_train_dist, how='left', on='id')
df_train_merge=df_train_merge.merge(df_train_brand, how='left', on='id')
print "Creating csv"
df_train_merge.to_csv("../../data/feat/train_distfeat_brandFeat_countFeat1.csv")