import pandas as pd

print "Reading files"
cols=['query', 'title', 'description', 'values']
df_test_dist=pd.read_csv("../../data/feat/test_distFeat.csv")
df_test_dist=df_test_dist.drop(cols, axis=1)
df_test_counting=pd.read_csv("../../data/feat/test_countingfeat_part1.csv")
#df_train_counting=df_train_counting.drop(cols, axis=1)
df_test_brand=pd.read_csv("../../data/feat/test_brandFeat.csv")
print "Merging files"
df_test_merge=df_test_counting.merge(df_test_dist, how='left', on='id')
df_test_merge=df_test_merge.merge(df_test_brand, how='left', on='id')
print "Creating csv"
df_test_merge.to_csv("../../data/feat/test_distfeat_brandFeat_countFeat1.csv")
print "Columns: ",df_test_merge.columns