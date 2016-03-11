from sklearn import cross_validation
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

df = pd.read_csv("../../data/train_only_features.csv")
relevance_train=df["relevance"]
df=df.drop('relevance', axis=1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df, relevance_train, test_size=0.3, random_state=0)
#etr = ExtraTreesRegressor(n_estimators=10, max_features=0.5, n_jobs=2,random_state=2015)
#etr = RandomForestRegressor(n_estimators=10, max_features=0.5, n_jobs=2,random_state=2015)
etr=LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                    fit_intercept=True, intercept_scaling=1.0,
                                    class_weight='auto', random_state=2015)
etr.fit(X_train, y_train)
pred = etr.predict_proba(X_test)                                            
rms = sqrt(mean_squared_error(y_test, pred))
print "Error: ", rms

#Result:
#Extra tree- 0.50
#Random forest-0.494