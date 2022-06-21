from pydantic import BaseModel, create_model
import dataAnalysis

#sample = dataAnalysis.loadData('featSelect_X',id= 100002, index = True).iloc[0,:].to_dict()
sample = dataAnalysis.loadData(table='featSelect',id= 100002, index = True).drop(columns=['SK_ID_CURR','TARGET']).to_dict('r')[0]

dynamicModel = create_model('dynamicModel',**sample)

# create a class with pydantic that will help structuring input data
class credit_application(dynamicModel):
    pass


