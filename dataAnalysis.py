import pandas as pd 
import requests

NGROK_URL = 'http://67b2-93-121-168-240.ngrok.io'

def loadData(table, id:int = -1, index = True):
    url = NGROK_URL+"/data?table="+table+"&id="+str(id)
    resp = requests.get(url)
    if index : 
        df = pd.read_json(resp.json(),orient ='records').set_index('index')
    else : 
        df = pd.read_json(resp.json(),orient ='records')
    return(df)

def loadDataIndexes(table, gender, family, id:int = -1, index = True):
    url = NGROK_URL+"/dataIndex?table="+table+"&id="+str(id)+"&gender="+str(gender)+"&family="+str(family)
    resp = requests.get(url)
    if index : 
        df = pd.read_json(resp.json(),orient ='records').set_index('index')
    else : 
        df = pd.read_json(resp.json(),orient ='records')
    return(df)