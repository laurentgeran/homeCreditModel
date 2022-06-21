import pandas as pd
import numpy as np
import scipy
import re

from sklearn.pipeline import Pipeline

# Preprocess
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

"""def localLoad(table,cursor):
    cursor.execute("SELECT * FROM "+table)
    columns = cursor.description 
    result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cursor.fetchall()]
    resultJSON = json.dumps(result)
    df = pd.read_json(resultJSON,orient ='records')
    return (df)"""

def localLoad(table,connection,index=False):
    query = "SELECT * FROM "+table
    if index:
        df = pd.read_sql(query, connection, index_col = 'index')
    else :
        df = pd.read_sql(query, connection)
    return (df)

def getVarByTypes(dataframe,listTypes:list):
    listVariables=list(dataframe.select_dtypes(include=listTypes).columns)
    return listVariables

def winsorize2(dataframe,variable,l=0.01):
    column=dataframe.loc[:,variable]
    mask=np.logical_not(np.ma.masked_invalid(column).mask).reshape(1,-1)[0]
    columnUnmasked=column.loc[mask]
    if len(columnUnmasked.value_counts())>3:
        variableWinsorized=winsorize(columnUnmasked,limits=[l,l])
        while len(np.unique(variableWinsorized))<3:
            l=l/2
            variableWinsorized = winsorize(columnUnmasked,limits=[l,l])
        dataframe.loc[mask,variable]=variableWinsorized
    else :
        #print("{} has too few values to be winsorized".format(variable))
        pass
    pass
 
def winsorizeAll(dataframe,idVar):
    dataframe.reset_index(drop=True, inplace=True)
    listVariables=getVarByTypes(dataframe,["int64",'float64'])
    listVariables=[var for var in listVariables if var not in idVar]
    for variable in listVariables:
        winsorize2(dataframe,variable)
    pass  

def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []
    #the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_[:-1]:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names_out()
        # if no 'get_feature_names' function, use raw column name
        except AttributeError: 
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name

def removeColinear(dataframe,numVar):
    corrs = dataframe.loc[:,numVar].corr()
    colinearIndexes=[]
    l=len(numVar)
    for i in range(l) :
        for j in range(i+1,l) : 
            if corrs.iloc[i,j]>0.85:
                miss_i = dataframe.loc[:,numVar[i]].isna().sum()
                miss_j = dataframe.loc[:,numVar[j]].isna().sum()
                if miss_i > miss_j :
                    colinearIndexes.append(i)
                else : 
                    colinearIndexes.append(j)
    var=[numVar[i] for i in range(l) if i not in colinearIndexes]
    return var

def removeDependent(dataframe,catVar):    
    dependentIndexes=[]
    l=len(catVar)
    for i in range(l) :
        for j in range(i+1,l) :
            contingencyTable = pd.crosstab(dataframe[catVar[i]],dataframe[catVar[j]])
            if (contingencyTable>5).all().all():
                chi2, p, dof, exp = scipy.stats.chi2_contingency(contingencyTable)
                if p<0.01:
                    miss_i = dataframe.loc[:,catVar[i]].isna().sum()
                    miss_j = dataframe.loc[:,catVar[j]].isna().sum()
                    if miss_i > miss_j :
                        dependentIndexes.append(i)
                    else : 
                        dependentIndexes.append(j)
                    
    var=[catVar[i] for i in range(l) if i not in dependentIndexes]
    return var


def preprocess(dataframe,aggregation=False,_id=np.nan):
    
    #define transformers
    categoricalTransformer = Pipeline (steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                            ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    numericalTransformer = Pipeline (steps=[('imputer', SimpleImputer(strategy='median'))])

    # Removing variables with too much missing values
    l= len(dataframe)
    dataframe = dataframe.loc[:,(dataframe.isna().sum()/l<0.5).values]
    
    # Removing individual with too much missing values
    l= len(dataframe.columns)
    dataframe = dataframe.loc[(dataframe.isna().sum(axis=1)/l<0.3),:]
    
    boolVar =  getVarByTypes(dataframe,["bool"])
    
    for var in boolVar:
        dataframe.loc[:,var]=dataframe[var].astype("int64")
    
    originalVar = list(dataframe.columns)
    
    catVar = getVarByTypes(dataframe,["object"])
    intVar = getVarByTypes(dataframe,["int64","float64"])    
    idBool = ["SK_ID" in col for col in intVar]
    numVar=[var for (var, remove) in zip(intVar, idBool) if not remove]
    idVar=[var for var in intVar if var not in numVar]
    
    # Removing colinear variables
    numVar = removeColinear(dataframe,numVar)
    catVar = removeDependent(dataframe,catVar)
               
    preprocessor = ColumnTransformer(transformers=[('cat', categoricalTransformer, catVar),
                                                   ( 'num', numericalTransformer, numVar)],
                                    remainder='drop')
    
    # Replacing some infinity values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    
    df = preprocessor.fit_transform(dataframe)
    dfId = dataframe.loc[:,idVar].reset_index(drop = True)
    newVar=get_column_names_from_ColumnTransformer(preprocessor)
    if isinstance(df, np.ndarray):
        df=pd.DataFrame(df,columns=newVar)
    else:
        df=pd.DataFrame(df.todense(),columns=newVar)
    df=pd.concat((df,dfId),axis=1)
    
    catVarEncoded = [c for c in newVar if c not in originalVar]
    
    winsorizeAll(df,idVar)

    if aggregation :
        agg={}
        for var in numVar:
            agg[var]=["min", 'max',"median","mean","sum"]
        for var in catVarEncoded:
            agg[var]=["count","sum","mean"]

        df=df.groupby(_id).agg(agg)
        df.columns = ['_'.join(col) for col in df.columns.values]
    else :
        pass  
    
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    #print(np.any(np.isnan(df)))
    #print(np.all(np.isfinite(df)))
    
    return (df)