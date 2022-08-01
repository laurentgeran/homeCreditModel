import numpy as np
import matplotlib.pyplot as plt

# Modeling 
from lightgbm import LGBMClassifier

# Scaling 
from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler

# Sampling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler 

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

# Hyperparameters optimization
from hyperopt import STATUS_OK


def modeling(classifier, X, y, sampling = 0, thresh = 0.5, plotPR = False):
    XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size=0.2)
    transformer = RobustScaler()
    XTrain = transformer.fit_transform(XTrain)
    XTest = transformer.transform(XTest)
    if sampling == 1 :
        oversampling = BorderlineSMOTE(sampling_strategy= 0.66, random_state=1)
        XTrain, yTrain = oversampling.fit_resample(XTrain, yTrain)
    elif sampling == -1 :
        randomundersampling = RandomUnderSampler(sampling_strategy= 0.66, random_state=1)
        XTrain, yTrain = randomundersampling.fit_resample(XTrain, yTrain)
    else : 
        pass
    auc_score = make_scorer(roc_auc_score)
    accuracy = make_scorer(accuracy_score)
    g_norm_score = make_scorer(customMetric)
    scoring = {'auc': auc_score, 'accuracy': accuracy, 'gain':g_norm_score}
    cv = cross_validate(classifier,XTrain, yTrain , cv=4,scoring = scoring)
    print('score auc : {score}'.format(score=np.mean(cv['test_auc'])))
    print('score accuracy : {score}'.format(score=np.mean(cv['test_accuracy'])))
    print('score gain train: {score}'.format(score=np.mean(cv['test_gain'])))

    model = classifier.fit(XTrain, yTrain)
    yPred_proba = classifier.predict_proba(XTest)
    yPred=[1 if x[1]>thresh else 0 for x in yPred_proba]
    yPred_proba_positive = yPred_proba[:, 1]
    
    # calculate inputs for the PR curve
    precision, recall, thresholds = precision_recall_curve(yTest, yPred_proba_positive)

    print(precision)
    print(recall)

    if plotPR :
        # calculate the no skill line as the proportion of the positive class
        no_skill = len(y[y==1]) / len(y)
        # plot the no skill precision-recall curve
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # plot PR curve
        plt.plot(recall, precision, marker='.', label='classifier')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        #plt.show(block=False)
        plt.show()

    # calculate and print PR AUC
    auc_pr = auc(recall, precision)

    # calculate and print Brier Score
    brier = brier_score_loss(yTest, yPred_proba_positive)
    
    
    print('score gain test : {score}'.format(score=g_norm_score(model,XTest,yTest)))
    print('AUC PR: %.3f' % auc_pr)
    print('Brier Score: %.3f' % (brier))
    print('classification report : ', classification_report(yTest,yPred))

    return model, XTrain, yTrain

def customMetric(y, y_pred, fn_value=-10, fp_value=-1, tp_value=10, tn_value=1):
    
    # Matrice de Confusion
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # gain total
    g = tp*tp_value + tn*tn_value + fp*fp_value + fn*fn_value
    # gain maximum
    g_max = (fp + tn)*tn_value + (fn + tp)*tp_value
    # gain minimum
    g_min = (fp + tn)*fp_value + (fn + tp)*fn_value
    # gain normalisé
    g_norm = (g - g_min)/(g_max - g_min)

    return g_norm

def hyperopt_train_test(params, X, y):
    
    # On s'assure que les paramètres soient au bon format
    for parameter_name in ['num_leaves','max_depth','n_estimators']:
        params[parameter_name] = int(params[parameter_name])

    # Paramètres du modèle    
    params_model = {'n_estimators': params['n_estimators'], 
                    'class_weight': params['class_weight'],
                    'max_depth': int(params['max_depth']), 
                    'learning_rate': params['learning_rate'],
                    'subsample': params['subsample'],
                    'num_leaves': int(params['num_leaves']),
                    'reg_alpha': params['reg_alpha'],
                    'reg_lambda': params['reg_lambda']
                   }
    thresh = params['solvability_threshold']
    
    skf = StratifiedKFold(n_splits=4)
    
    clf = LGBMClassifier(**params_model)

    g_norm_skf = []

    for train_index, test_index in skf.split(X, y):
        clf.fit(X.loc[train_index],y[train_index])
        proba = clf.predict_proba(X.loc[test_index])
        y_pred=[1 if x[1]>thresh else 0 for x in proba]
        g_norm_skf.append(customMetric(y[test_index], y_pred))
        
    g_norm = np.mean(g_norm_skf)
    
    loss = 1 - g_norm
    
    return loss