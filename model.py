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

model = modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y)

# sampling
model_under = modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y, sampling = -1)
model_over = modelFunc.modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y, sampling = 1)

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

modelFinal = modelFunc.modeling(LGBMClassifier(), featSelect_X, applicationTrain_y, sampling = 0)
modelFinal_under = modelFunc.modeling(LGBMClassifier(), featSelect_X, applicationTrain_y, sampling = -1)
modelFinal_over = modelFunc.modeling(LGBMClassifier(), featSelect_X, applicationTrain_y, sampling = 1)

# optimisation des hyperparamètres

# saving the model
with open('homecredit_model/clf_feat_under.pkl', 'wb') as output_file:
    pickle.dump(modelFinal_under[0], output_file)

con.close()