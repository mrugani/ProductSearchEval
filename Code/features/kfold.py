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

df = pd.read_csv("../../data/feat/train_final1.csv")
relevance_train=df["relevance"]
df=df.drop([
	'jaccard_coef_of_trigram_between_query_title',
    'jaccard_coef_of_trigram_between_query_description',
    'jaccard_coef_of_trigram_between_query_attribute_values',
    'jaccard_coef_of_trigram_between_title_description',
    'jaccard_coef_of_trigram_between_title_attribute_values',
    'jaccard_coef_of_trigram_between_description_attribute_values',
    'dice_dist_of_trigram_between_query_title',
    'dice_dist_of_trigram_between_query_description',
    'dice_dist_of_trigram_between_query_attribute_values',
    'dice_dist_of_trigram_between_title_description',
    'dice_dist_of_trigram_between_title_attribute_values',
    'dice_dist_of_trigram_between_description_attribute_values',
	'relevance',
    'id',
], axis=1)
test = pd.read_csv("../../data/feat/test_allfeat.csv")
test["pid"]=test["pid_x"]
test=test.drop([
	'jaccard_coef_of_trigram_between_query_title',
    'jaccard_coef_of_trigram_between_query_description',
    'jaccard_coef_of_trigram_between_query_attribute_values',
    'jaccard_coef_of_trigram_between_title_description',
    'jaccard_coef_of_trigram_between_title_attribute_values',
    'jaccard_coef_of_trigram_between_description_attribute_values',
    'dice_dist_of_trigram_between_query_title',
    'dice_dist_of_trigram_between_query_description',
    'dice_dist_of_trigram_between_query_attribute_values',
    'dice_dist_of_trigram_between_title_description',
    'dice_dist_of_trigram_between_title_attribute_values',
    'dice_dist_of_trigram_between_description_attribute_values',
    'pid_x',
 'pid_y.1',
 'pid_y'
], axis=1)
train_base=xgb.DMatrix(df, label=relevance_train)
#params = {'num_round': 5.0, 'task': 'regression', 'colsample_bytree': 1.0, 'silent': 1, 'nthread': 1, 'min_child_weight': 2.0, 'subsample': 0.8, 'eta': 0.43, 'objective': 'reg:linear', 'max_evals': 200, 'seed': 113, 'max_depth': 6, 'gamma': 1.4000000000000001, 'booster': 'gbtree'}
params =  {'num_round': 5.0, 'task': 'regression', 'colsample_bytree': 0.6000000000000001, 'silent': 1, 'nthread': 1, 'min_child_weight': 1.0, 'subsample': 1.0, 'eta': 0.35000000000000003, 'objective': 'reg:linear', 'max_evals': 200, 'seed': 111, 'max_depth': 6, 'gamma': 1.1, 'booster': 'gbtree'}
id_test=test['id']
test=test.drop('id', axis=1)
print "train", df.shape
print "test", test.shape
gbm = xgb.train(dtrain=train_base,params=params)
pred = gbm.predict(xgb.DMatrix(test))
#rms = sqrt(mean_squared_error(y_test, pred))
for i in range(len(pred)):
    if pred[i]<1.0:
        pred[i] = 1.0
    if pred[i]>3.0:
        pred[i] = 3.0
pd.DataFrame({"id": id_test, "relevance": pred}).to_csv('submission_after.csv',index=False)
