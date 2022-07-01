import uvicorn 
import pickle
import numpy as np
from fastapi import FastAPI
import shap
import json

import dataAnalysis
import creditApp

# create the app
app = FastAPI()

# load model
with open(r"clf_feat_under.pkl", "rb") as input_file:
    model = pickle.load(input_file)

@app.get('/')
def home():
    return({'message':'Hello world !'})

@app.get('/shap')
def get_shap(SK_ID_CURR):
    explainer = shap.TreeExplainer(model)

    featSelect = dataAnalysis.loadData(table= 'featSelect',index=True)
    index = featSelect.index[featSelect['SK_ID_CURR'] == int(SK_ID_CURR)].values[0]
    featSelect = featSelect.drop(columns=['SK_ID_CURR','TARGET'])

    shap_values = explainer.shap_values(featSelect)

    varValues=[]
    for i,var in enumerate(featSelect.columns):
        if shap_values[1][index][i]>0:
            varValues.append((var,shap_values[1][index][i]))
    varValues = sorted(varValues, key=lambda tup: tup[1], reverse = True)
    
    response = [shap_values[1][index].tolist()]
    response.append(featSelect.columns.values.tolist())
    response.append(varValues)

    print(response)

    return(json.dumps(response))

@app.post('/predict')
def predict_score(data:creditApp.credit_application):
    received=data.dict()
    var = np.array([received[x] for x in received.keys()]).reshape(1, -1)
    score = model.predict_proba(var)[0][1]
    return({'prediction': score})

# run the api
if __name__ == '__main__':
    uvicorn.run('modelAPI:app' , host='127.0.0.1', port=4000, debug=True)
#uvicorn  modelAPI:app --reload --port 4000 


