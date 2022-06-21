import sqlite3
import pandas as pd
import gc
import preproFunc

# creating file path
DB_FILE = 'D:\projects\openclassrooms\projets\P7_geran_laurent\homecredit_data\db.db'

# creating cursor
con = sqlite3.connect(DB_FILE)
#cur = con.cursor()

applicationTrain = preproFunc.localLoad('application_train', con)
#applicationTest = preproFunc.localLoad('application_test', con)

previousApplication = preproFunc.localLoad('previous_application', con)

# posCashBalance
posCashBalance = preproFunc.localLoad('POS_CASH_balance', con)

posCashBalance['LATE_PAYMENT'] = posCashBalance['SK_DPD'] > 0.0
posCashBalance['INSTALLMENTS_PAID'] = posCashBalance['CNT_INSTALMENT'] - posCashBalance['CNT_INSTALMENT_FUTURE']

previousApplication = pd.merge(previousApplication,preproFunc.preprocess(posCashBalance,aggregation=True,_id="SK_ID_PREV"), how = "left",left_on="SK_ID_PREV",right_on="SK_ID_PREV",suffixes=("_prev","_pos"))

del posCashBalance
gc.collect()

# instalmentsPayments
instalmentsPayments = preproFunc.localLoad('installments_payments', con)

instalmentsPayments['LATE'] = instalmentsPayments['DAYS_ENTRY_PAYMENT'] > instalmentsPayments['DAYS_INSTALMENT']
instalmentsPayments['LOW_PAYMENT'] = instalmentsPayments['AMT_PAYMENT'] < instalmentsPayments['AMT_INSTALMENT']

previousApplication = pd.merge(previousApplication,preproFunc.preprocess(instalmentsPayments,aggregation=True,_id="SK_ID_PREV"), how = "left",left_on="SK_ID_PREV",right_on="SK_ID_PREV",suffixes=("_prev","_instal"))

del instalmentsPayments
gc.collect()

# creditCardBalance
creditCardBalance = preproFunc.localLoad('credit_card_balance', con)

creditCardBalance['OVER_LIMIT'] = creditCardBalance['AMT_BALANCE'] > creditCardBalance['AMT_CREDIT_LIMIT_ACTUAL']
creditCardBalance['BALANCE_CLEARED'] = creditCardBalance['AMT_BALANCE'] == 0.0
creditCardBalance['LOW_PAYMENT'] = creditCardBalance['AMT_PAYMENT_CURRENT'] < creditCardBalance['AMT_INST_MIN_REGULARITY']
creditCardBalance['LATE'] = creditCardBalance['SK_DPD'] > 0.0

previousApplication = pd.merge(previousApplication,preproFunc.preprocess(creditCardBalance,aggregation=True,_id="SK_ID_PREV"), how = "left",left_on="SK_ID_PREV",right_on="SK_ID_PREV",suffixes=("_prev","_cred"))

del creditCardBalance
gc.collect()

# previousApplication
previousAppCounts = previousApplication.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(columns = {'SK_ID_PREV': 'previousAppCounts'})
applicationTrain = applicationTrain.merge(previousAppCounts, on = 'SK_ID_CURR', how = 'left')
# Fill the missing values with 0 
applicationTrain['previousAppCounts'] = applicationTrain['previousAppCounts'].fillna(0)

applicationTrain = pd.merge(applicationTrain,preproFunc.preprocess(previousApplication,aggregation=True,_id="SK_ID_CURR"), how = "left",left_on="SK_ID_CURR",right_on="SK_ID_CURR",suffixes=("_curr","_prev"))

del previousApplication
gc.collect()

##
bureau = preproFunc.localLoad('bureau', con)

bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']

# bureauBalance
bureauBalance = preproFunc.localLoad('bureau_balance', con)

bureauBalance['PAST_DUE'] = bureauBalance['STATUS'].isin(['1', '2', '3', '4', '5'])
bureauBalance['ON_TIME'] = bureauBalance['STATUS'] == '0'

bureau = pd.merge(bureau,preproFunc.preprocess(bureauBalance,aggregation=True,_id="SK_ID_BUREAU"), how = "left",left_on="SK_ID_BUREAU",right_on="SK_ID_BUREAU",suffixes=("_bur","_burbal"))

del bureauBalance
gc.collect()

# bureau
previousLoanCounts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previousLoanCounts'})
applicationTrain = applicationTrain.merge(previousLoanCounts, on = 'SK_ID_CURR', how = 'left')
# Fill the missing values with 0 
applicationTrain['previousLoanCounts'] = applicationTrain['previousLoanCounts'].fillna(0)

applicationTrain = pd.merge(applicationTrain,preproFunc.preprocess(bureau,aggregation=True,_id="SK_ID_CURR"), how = "left",left_on="SK_ID_CURR",right_on="SK_ID_CURR",suffixes=("_curr","_bureau"))

del bureau
gc.collect()

# applicationTrain
applicationTrain['LOAN_RATE'] = applicationTrain['AMT_ANNUITY'] / applicationTrain['AMT_CREDIT'] 
applicationTrain['CREDIT_INCOME_RATIO'] = applicationTrain['AMT_CREDIT'] / applicationTrain['AMT_INCOME_TOTAL']
applicationTrain['EMPLOYED_BIRTH_RATIO'] = applicationTrain['DAYS_EMPLOYED'] / applicationTrain['DAYS_BIRTH']
applicationTrain['EXT_SOURCE_SUM'] = applicationTrain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
applicationTrain['EXT_SOURCE_MEAN'] = applicationTrain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
applicationTrain['AMT_REQ_SUM'] = applicationTrain[[x for x in applicationTrain.columns if 'AMT_REQ_' in x]].sum(axis = 1)

applicationTrain=preproFunc.preprocess(applicationTrain)
applicationTrain_y=applicationTrain["TARGET"]
applicationTrain_X=applicationTrain.drop(columns=["TARGET"])

applicationTrain.to_sql('applicationTrain', con)
applicationTrain_y.to_sql('applicationTrain_y', con)
applicationTrain_X.to_sql('applicationTrain_X', con)

con.close()

#columns = cur.description 
#result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur.fetchall()]
# Be sure to close the connection
