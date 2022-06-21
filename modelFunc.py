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

# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

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

def modeling2(classifier, X, y, featuresSelected:list= ["All"], overSampling = False):
    
    X_=X
    if featuresSelected[0] != "All":
        X_=X_.loc[:,featuresSelected]
    XTrain, XTest, yTrain, yTest=splitData(X_,y)
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