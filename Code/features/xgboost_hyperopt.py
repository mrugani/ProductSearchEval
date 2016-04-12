from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

train=pd.read_csv("../../data/train_distFeat_updated_10.csv")
train7=pd.read_csv("../../data/train_distFeat_updated_7.csv")
train["lzma"]=train7["lzma"]
train8=pd.read_csv("../../data/train_distFeat_updated_8.csv")
train["last_word_in"]=train8["last_word_in"]
train["edist"]=train8["edist"]
train=train.replace(np.nan,0, regex=True)
y=train['relevance']
# train = train.drop('id', axis=1)
df1=pd.read_csv('../../data/train_brand_dist_counting_cosine_values.csv')
train['cosine_sim_query_title']=df1['cosine_sim_query_title']
train['cosine_sim_query_description']=df1['cosine_sim_query_description']
train['cosine_sim_title_description']=df1['cosine_sim_title_description']
#train['cos']=train['cosine_sim_query_title']+train['cosine_sim_query_description']-2*train['cosine_sim_title_description']
train=train.drop([  
    'dice_dist_of_trigram_between_query_title',
    'dice_dist_of_trigram_between_query_description',
    'dice_dist_of_trigram_between_query_attribute_values',
    'dice_dist_of_trigram_between_title_description',
    'dice_dist_of_trigram_between_title_attribute_values',
    'dice_dist_of_trigram_between_description_attribute_values',
  'pos_of_attribute_values_trigram_in_description_min',
 'pos_of_attribute_values_trigram_in_description_mean',
 'pos_of_attribute_values_trigram_in_description_median',
 'pos_of_attribute_values_trigram_in_description_max',
 'pos_of_attribute_values_trigram_in_description_std',
 'normalized_pos_of_attribute_values_trigram_in_description_min',
 'normalized_pos_of_attribute_values_trigram_in_description_mean',
 'normalized_pos_of_attribute_values_trigram_in_description_median',
 'normalized_pos_of_attribute_values_trigram_in_description_max',
 'normalized_pos_of_attribute_values_trigram_in_description_std',
], axis=1)
train = train.drop('relevance', axis=1)

def score(params):
    print "Training with params : "
    print params
    #num_round = int(params['n_estimators'])
    #del params['n_estimators']
    params['max_depth']=int(params['max_depth'])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain)
    pred = model.predict(dvalid)
    for i in range(len(pred)):
	    if pred[i]<1.0:
	        pred[i] = 1.0
	    if pred[i]>3.0:
	        pred[i] = 3.0
    rms=sqrt(mean_squared_error(y_test, pred))
    print "\tScore {0}\n\n".format(rms)
    return {'loss': rms, 'status': STATUS_OK}

def score_rf(params):
	rfr = RandomForestRegressor(n_estimators = 10, n_jobs = -1, random_state = 2016)
	rfr.fit(X_train, y_train)
	pred = rfr.predict(X_test)
	rms=sqrt(mean_squared_error(y_test, pred))
	print "\tScore {0}\n\n".format(rms) 
	return {'loss': rms, 'status': STATUS_OK}

params = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', 5, 10, 5),
    'nthread': 1,
    'silent': 1,
    'seed': 111,
    "max_evals": 200,
}

# params={'task': 'regression',
#     'booster': 'gblinear',
#     'objective': 'reg:linear',
#     'eta' : hp.quniform('eta', 0.01, 1, 0.01),
#     'lambda' : hp.quniform('lambda', 0, 5, 0.05),
#     'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
#     'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
#     'num_round' : hp.quniform('num_round', 5, 10, 5),
#     'nthread': 1,
#     'silent' : 1,
#     'seed': 191,
#     "max_evals": 200}

print "Splitting data into train and valid ...\n\n"
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=1234)
trials = Trials()
best = fmin(score, params, algo=tpe.suggest, trials=trials, max_evals=250)
