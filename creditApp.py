import sqlite3
from pydantic import BaseModel, create_model
import preproFunc

DB_FILE = 'D:\projects\openclassrooms\projets\P7_geran_laurent\homecredit_data\db.db'

# creating cursor
con = sqlite3.connect(DB_FILE)

sample = preproFunc.localLoad('featSelect_X',con, index = True).iloc[0,:].to_dict()

dynamicModel = create_model('dynamicModel',**sample)

# create a class with pydantic that will help structuring input data
class credit_application(dynamicModel):
    pass


