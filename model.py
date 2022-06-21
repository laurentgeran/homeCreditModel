import pandas as pd
import numpy as np
import sqlite3
import preproFunc
import modelFunc
import warnings

# Modeling
from lightgbm import LGBMClassifier

# Hyperparameters 
from skopt.utils import use_named_args
from skopt import gp_minimize
 
# Feature selection
from boruta import BorutaPy

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

modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y)

#with open('clf.pkl', 'wb') as output_file:
    #pickle.dump(model, output_file)

# random under sampling
n = len(applicationTrain[applicationTrain["TARGET"]==1])
dataBaseline = applicationTrain[applicationTrain["TARGET"]==0].sample(n,random_state=0,axis=0)
dataBaseline=dataBaseline.append(applicationTrain[applicationTrain["TARGET"]==1])
under_y=dataBaseline["TARGET"]
under_X=dataBaseline.drop(columns=["TARGET","SK_ID_CURR"])

modelFunc.modeling(LGBMClassifier(), under_X, under_y)

# feature selection with BorutaPy
feat_selector = BorutaPy(LGBMClassifier(num_boost_round=15), n_estimators='auto', random_state=1)
feat_selector.fit(under_X.values, under_y.values)
feat_selected = applicationTrain_X.columns[feat_selector.support_].values.tolist()
feat_selected.extend(["SK_ID_CURR", "TARGET"])


# oversampling using feature selection
featSelect=applicationTrain.loc[:,feat_selected]
featSelect_y = featSelect["TARGET"]
featSelect_X = featSelect.drop(columns=["TARGET","SK_ID_CURR"])

con.cursor().execute("DROP TABLE IF EXISTS featSelect")
con.cursor().execute("DROP TABLE IF EXISTS featSelect_y")
con.cursor().execute("DROP TABLE IF EXISTS featSelect_X")

featSelect.to_sql('featSelect', con)
featSelect_y.to_sql('featSelect_y', con)
featSelect_X.to_sql('featSelect_X', con)

modelOver = modelFunc.modeling(LGBMClassifier(), featSelect_X, applicationTrain_y, overSampling = True)

# optimisation des hyperparamètres

# saving the model
with open('homecredit_model/clf_feat_over.pkl', 'wb') as output_file:
    pickle.dump(modelOver, output_file)


con.close()