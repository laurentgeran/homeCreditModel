import uvicorn 
import pickle
import numpy as np
from fastapi import FastAPI

import creditApp

# create the app
app = FastAPI()

# load model
with open(r"clf_feat_over.pkl", "rb") as input_file:
    model = pickle.load(input_file)

@app.get('/')
def home():
    return({'message':'Hello world !'})

@app.post('/predict')
def predict_score(data:creditApp.credit_application):
    received=data.dict()
    var = np.array([received[x] for x in received.keys()]).reshape(1, -1)
    score = model.predict_proba(var)[0][1]
    return({'prediction': score})

# run the api
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)

#uvicorn  modelAPI:app --reload


