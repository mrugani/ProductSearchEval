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

df = pd.read_csv("../../data/train_distFeat_updated_6.csv")
df=df.replace(np.nan,0, regex=True)
relevance_train=df["relevance"]
df=df.drop('relevance', axis=1)
df=df.drop('id', axis=1)
df=df.drop([
 'dice_dist_of_trigram_between_query_title',
    'dice_dist_of_trigram_between_query_description',
    'dice_dist_of_trigram_between_query_attribute_values',
    'dice_dist_of_trigram_between_title_description',
    'dice_dist_of_trigram_between_title_attribute_values',
    'dice_dist_of_trigram_between_description_attribute_values',
 'pos_of_title_trigram_in_description_min',
 'pos_of_title_trigram_in_description_mean',
 'pos_of_title_trigram_in_description_median',
 'pos_of_title_trigram_in_description_max',
 'pos_of_title_trigram_in_description_std',
 'normalized_pos_of_title_trigram_in_description_min',
 'normalized_pos_of_title_trigram_in_description_mean',
 'normalized_pos_of_title_trigram_in_description_median',
 'normalized_pos_of_title_trigram_in_description_max',
 'normalized_pos_of_title_trigram_in_description_std',
 
], axis=1)
df1=pd.read_csv('../../data/train_brand_dist_counting_cosine_values.csv')
df['cosine_sim_query_title']=df1['cosine_sim_query_title']
df['cosine_sim_query_description']=df1['cosine_sim_query_description']
df['cosine_sim_title_description']=df1['cosine_sim_title_description']
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(df, relevance_train, test_size=0.3, random_state=0)
#etr = ExtraTreesRegressor(n_estimators=10, max_features=0.5, n_jobs=2,random_state=2015)
# etr = RandomForestRegressor(n_estimators=10, max_features=0.5, n_jobs=2,random_state=2015)
# # etr=LogisticRegression(penalty="l2", dual=True, tol=1e-5,
# #                                     fit_intercept=True, intercept_scaling=1.0,
# #                                     class_weight='auto', random_state=2015)
# etr.fit(df, relevance_train)
test = pd.read_csv("../../data/test_distFeat_updated_6.csv")
df1=pd.read_csv('../../data/test_dist_counting_cosine.csv')
test['cosine_sim_query_title']=df1['cosine_sim_query_title']
test['cosine_sim_query_description']=df1['cosine_sim_query_description']
test['cosine_sim_title_description']=df1['cosine_sim_title_description']
test=test.drop([
 'dice_dist_of_trigram_between_query_title',
    'dice_dist_of_trigram_between_query_description',
    'dice_dist_of_trigram_between_query_attribute_values',
    'dice_dist_of_trigram_between_title_description',
    'dice_dist_of_trigram_between_title_attribute_values',
    'dice_dist_of_trigram_between_description_attribute_values',
 'pos_of_title_trigram_in_description_min',
 'pos_of_title_trigram_in_description_mean',
 'pos_of_title_trigram_in_description_median',
 'pos_of_title_trigram_in_description_max',
 'pos_of_title_trigram_in_description_std',
 'normalized_pos_of_title_trigram_in_description_min',
 'normalized_pos_of_title_trigram_in_description_mean',
 'normalized_pos_of_title_trigram_in_description_median',
 'normalized_pos_of_title_trigram_in_description_max',
 'normalized_pos_of_title_trigram_in_description_std',
], axis=1)

test=test.replace(np.nan,0, regex=True)
# pred = etr.predict(test)                                            
# #rms = sqrt(mean_squared_error(y_test, pred))
# id_test=test['id']
# for i in range(len(pred)):
#     if pred[i]<1.0:
#         pred[i] = 1.0
#     if pred[i]>3.0:
#         pred[i] = 3.0
# pd.DataFrame({"id": id_test, "relevance": pred}).to_csv('submission_after.csv',index=False)
#Result:
#Extra tree- 0.50
#Random forest-0.494
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
