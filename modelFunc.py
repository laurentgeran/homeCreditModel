import numpy as np

# Modeling 
from lightgbm import LGBMClassifier

# Scaling 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# Sampling
from imblearn.over_sampling import SMOTE 

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

# Hyperparameters optimization
from hyperopt import hp
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import STATUS_OK


def splitData(X,y):
    XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size=0.2)
    return XTrain, XTest, yTrain, yTest

def modeling(classifier, X, y, overSampling = False):
    XTrain, XTest, yTrain, yTest=splitData(X,y)
    transformer = RobustScaler()
    XTrain = transformer.fit_transform(XTrain)
    XTest = transformer.transform(XTest)
    if overSampling:
        oversampling = SMOTE()
        XTrain, yTrain = oversampling.fit_resample(XTrain, yTrain)
    scoring = {"auc": make_scorer(roc_auc_score), "accuracy": make_scorer(accuracy_score)}
    cv = cross_validate(classifier,XTrain, yTrain , cv=4,scoring = scoring)
    #print(np.mean(cv["test_auc"]))
    #print(np.mean(cv["test_accuracy"]))
    classifier.fit(XTrain, yTrain,eval_metric='auc', eval_set=[(XTest, yTest)])
    yPred=classifier.predict(XTest)
    print("classification report : ", classification_report(yTest,yPred))

    return classifier

def customMetric(y_true, y_pred, fn_value=-100, fp_value=-1, tp_value=10, tn_value=1):
    
    # Matrice de Confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # gain total
    g = tp*tp_value + tn*tn_value + fp*fp_value + fn*fn_value
    # gain maximum
    g_max = (fp + tn)*tn_value + (fn + tp)*tp_value
    # gain minimum
    g_min = (fp + tn)*fp_value + (fn + tp)*fn_value
    # gain normalisé
    g_norm = (g - g_min)/(g_max - g_min)

    return g_norm

def hyperopt_train_test(params, applicationTrain_X, applicationTrain_y):
    
    # On s'assure que les paramètres soient au bon format
    for parameter_name in ['num_leaves','max_depth','n_estimators']:
        params[parameter_name] = int(params[parameter_name])

    # Paramètres du modèle    
    params_model = {'n_estimators': params['n_estimators'], 
                    'class_weight': params['class_weight'],
                    'max_depth': int(params['max_depth']), 
                    'learning_rate': params['learning_rate'],
                    'subsample': params['subsample'],
                    'colsample_bytree': params['colsample_bytree'],
                    'num_leaves': int(params['num_leaves']),
                    'reg_alpha': params['reg_alpha'],
                    'reg_lambda': params['reg_lambda']
                   }

    X_ = applicationTrain_X[:]
    
    skf = StratifiedKFold(n_splits=3)
    
    clf = LGBMClassifier(**params_model)
    
    y_pred = cross_val_predict(clf, X_, applicationTrain_y, method='predict', cv=skf)
    
    score = customMetric(applicationTrain_y, y_pred)
    
    loss = 1 - score
    
    return loss

space = {
    'n_estimators': hp.quniform('n_estimators', 200, 800, 200),
    #'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'max_depth' : hp.quniform('max_depth', 2, 30, 2),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'subsample': hp.quniform('subsample', 0.1, 1.0, 0.2),
    #'colsample_bytree': hp.quniform('colsample_by_tree', 0.6, 1.0, 0.1),
    'num_leaves': hp.quniform('num_leaves', 4, 100, 4),
    'reg_alpha': hp.quniform('reg_alpha', 0.1, 1.0, 0.1),
    'reg_lambda': hp.quniform('reg_lambda', 0.1, 1.0, 0.1),
    'solvability_threshold': hp.quniform('solvability_threshold', 0.0, 1.0, 0.025)
}


best = 1

def f(params):
    global best
    loss = hyperopt_train_test(params)
    if loss < best:
        best = loss
        print ('new best:', best, params)
    return {'loss': loss, 'status': STATUS_OK}

"""
trials = Trials()


best = fmin(f, space, algo=tpe.suggest, max_evals=5, trials=trials)

print('best:')
print(best)

params = {'learning_rate': 0.06830366370859095, 'max_depth': 22, 'n_estimators': 400, 'num_leaves': 24, 'reg_alpha': 0.4, 'reg_lambda': 0.7000000000000001, 'subsample': 0.6000000000000001}

model = modeling(LGBMClassifier(**params), applicationTrain_X, applicationTrain_y)

y_pred = model.predict(applicationTrain_X)

confusion_matrix(applicationTrain_y,y_pred)
"""