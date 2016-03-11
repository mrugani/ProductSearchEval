import pandas as pd

df=pd.read_csv("../../data/train_only_features_1.csv")
df=df.drop(["title"], axis=1)
df.to_csv("../../data/train_only_features.csv",header=True, index=False)