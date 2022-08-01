from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
import sqlite3
import preproFunc
import modelFunc
import warnings

# Modeling
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Hyperparameters 
from skopt.utils import use_named_args
from skopt import gp_minimize
 
# Feature selection
from boruta import BorutaPy

# Metrics
from sklearn.metrics import confusion_matrix

# Hyperparameters optimization
from hyperopt import hp
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import STATUS_OK


# Serialization
import pickle



# don't display warnings
warnings.filterwarnings("ignore")

# creating file path
DB_FILE = 'D:\projects\openclassrooms\projets\P7_geran_laurent\homecredit_data\db.db'

# creating cursor
con = sqlite3.connect(DB_FILE)
#cur = con.cursor()

applicationTrain = preproFunc.localLoad('applicationTrain', con, True)
applicationTrain_y= np.ravel(preproFunc.localLoad('applicationTrain_y', con, True))
applicationTrain_X= preproFunc.localLoad('applicationTrain_X', con, True).drop(columns=["SK_ID_CURR"])

print("NO SAMPLING :")
model = modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y)
model_sgd = modelFunc.modeling(SGDClassifier(loss='modified_huber'), applicationTrain_X, applicationTrain_y)

# sampling
print("UNDERSAMPLING :")
model_under = modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y, sampling = -1)
model_under_sgd = modelFunc.modeling(SGDClassifier(loss='modified_huber'), applicationTrain_X, applicationTrain_y, sampling = -1)

print("OVERSAMPLING :")
model_over = modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y, sampling = 1)
model_over_sgd = modelFunc.modeling(SGDClassifier(loss='modified_huber'), applicationTrain_X, applicationTrain_y, sampling = 1)

# feature selection with BorutaPy
feat_selector = BorutaPy(LGBMClassifier(num_boost_round=100), n_estimators='auto', random_state=1)
feat_selector.fit(model_under[1], model_under[2])
feat_selected = applicationTrain_X.columns[feat_selector.support_].values.tolist()
feat_selected.extend(["SK_ID_CURR", "TARGET"])


# sampling using feature selection
featSelect=applicationTrain.loc[:,feat_selected]
featSelect_y = featSelect["TARGET"]
featSelect_X = featSelect.drop(columns=["TARGET","SK_ID_CURR"])

con.cursor().execute("DROP TABLE IF EXISTS featSelect")
con.cursor().execute("DROP TABLE IF EXISTS featSelect_y")
con.cursor().execute("DROP TABLE IF EXISTS featSelect_X")

featSelect.to_sql('featSelect', con)
featSelect_y.to_sql('featSelect_y', con)
featSelect_X.to_sql('featSelect_X', con)

model_under = modelFunc.modeling(LGBMClassifier(), featSelect_X, featSelect_y, sampling = -1)

# optimisation des hyperparamètres
space = {
    'n_estimators': hp.quniform('n_estimators', 200, 800, 200),
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'max_depth' : hp.quniform('max_depth', 2, 30, 2),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'subsample': hp.quniform('subsample', 0.1, 1.0, 0.2),
    'num_leaves': hp.quniform('num_leaves', 4, 100, 4),
    'reg_alpha': hp.quniform('reg_alpha', 0.1, 1.0, 0.1),
    'reg_lambda': hp.quniform('reg_lambda', 0.1, 1.0, 0.1),
    'solvability_threshold': hp.quniform('solvability_threshold', 0.0, 1.0, 0.025)
}

def f(params):
    best = 1
    loss = modelFunc.hyperopt_train_test(params, featSelect_X, featSelect_y)
    if loss < best:
        best = loss
        print ('new best:', best, params)
    return {'loss': loss, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=5, trials=trials)

print('best:')

if best['class_weight']==1:
    best['class_weight']='balanced'
else:
    best['class_weight']=None

print(best)

paramsFinal = {
    'n_estimators': int(best['n_estimators']),
    'class_weight': best['class_weight'],
    'max_depth': int(best['max_depth']), 
    'learning_rate': best['learning_rate'],
    'subsample': best['subsample'],
    'num_leaves': int(best['num_leaves']),
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda']
}
threshFinal = best['solvability_threshold']

modelFinal = modelFunc.modeling(LGBMClassifier(**paramsFinal), featSelect_X, featSelect_y, sampling = -1, thresh = threshFinal)

#y_pred = modelFinal[0].predict(featSelect_X)

#print(confusion_matrix(featSelect_y,y_pred))

# saving the model
with open('homecredit_model/clf_feat_under.pkl', 'wb') as output_file:
    pickle.dump(modelFinal[0], output_file)

con.close()

#{'class_weight': 'balanced', 'learning_rate': 0.04322399776800654, 'max_depth': 28.0, 'n_estimators': 400.0, 'num_leaves': 52.0, 'reg_alpha': 0.4, 'reg_lambda': 0.30000000000000004, 'solvability_threshold': 0.35000000000000003, 'subsample': 1.0}