from sklearn import cross_validation
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import xgboost as xgb
from hyperopt import hp

df = pd.read_csv("../../data/feat/train_distfeat_brandFeat.csv")
relevance_train=df["relevance_x"]
df=df.drop([
	'relevance_x',
    'relevance_y',
    'id',
 'Unnamed: 0',
 'pid_y'
], axis=1)
test = pd.read_csv("../../data/feat/test_distfeat_brandFeat.csv")
test=test.drop([
 'Unnamed: 0.1',
 'Unnamed: 0',
 'pid_y'
], axis=1)
train_base=xgb.DMatrix(df, label=relevance_train)
params = {'num_round': 5.0, 'task': 'regression', 'colsample_bytree': 1.0, 'silent': 1, 'nthread': 1, 'min_child_weight': 2.0, 'subsample': 0.8, 'eta': 0.43, 'objective': 'reg:linear', 'max_evals': 200, 'seed': 113, 'max_depth': 6, 'gamma': 1.4000000000000001, 'booster': 'gbtree'}
id_test=test['id']
test=test.drop('id', axis=1)
gbm = xgb.train(dtrain=train_base,params=params)
pred = gbm.predict(xgb.DMatrix(test))
#rms = sqrt(mean_squared_error(y_test, pred))
for i in range(len(pred)):
    if pred[i]<1.0:
        pred[i] = 1.0
    if pred[i]>3.0:
        pred[i] = 3.0
pd.DataFrame({"id": id_test, "relevance": pred}).to_csv('submission_after.csv',index=False)
