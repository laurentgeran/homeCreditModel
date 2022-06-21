import requests
import json

url = "http://127.0.0.1:4000/predict"

payload = json.dumps({'x0_Cashloans': 1.0, 'x1_F': 0.0, 'x2_N': 1.0, 'x3_Working': 1.0, 
'AMT_CREDIT': 406597.5, 'AMT_ANNUITY': 24700.5, 'DAYS_BIRTH': -9461.0, 'DAYS_EMPLOYED': -637.0, 
'DAYS_REGISTRATION': -3648.0, 'EXT_SOURCE_2': 0.2629485927471776, 'EXT_SOURCE_3': 0.13937578009978951, 
'YEARS_BEGINEXPLUATATION_AVG': 0.9722, 'FLAG_DOCUMENT_3': 1.0, 'AMT_ANNUITY_median': 9251.775, 'AMT_ANNUITY_sum': 9251.775, 
'AMT_APPLICATION_median': 179055.0, 'DAYS_DECISION_min': -606.0, 'DAYS_DECISION_median': -606.0, 
'SELLERPLACE_AREA_sum': 500.0, 'CNT_PAYMENT_max': 24.0, 'CNT_PAYMENT_median': 24.0, 
'CNT_INSTALMENT_min_max': 24.0, 'SK_DPD_DEF_max_prev_median': 0.0, 'NUM_INSTALMENT_NUMBER_max_max': 19.0, 
'AMT_INSTALMENT_min_min': 9251.775, 'AMT_INSTALMENT_min_max': 9251.775, 'AMT_INSTALMENT_max_sum': 53093.745, 
'AMT_INSTALMENT_sum_max': 219625.69499999992, 'LATE_max_prev_median': 0.0, 'LATE_mean_prev_min': 0.0, 
'LATE_mean_prev_max': 0.0, 'LATE_mean_prev_median': 0.0, 'LOW_PAYMENT_mean_prev_min': 0.0, 
'LOW_PAYMENT_mean_prev_median': 0.0, 'x0_Cashloans_mean': 0.0, 'x0_Consumerloans_sum': 1.0, 
'DAYS_CREDIT_min': -1437.0, 'DAYS_CREDIT_ENDDATE_max': 780.0, 'DAYS_CREDIT_ENDDATE_mean': -344.25, 
'AMT_CREDIT_SUM_max': 450000.0, 'x2_Microloan_sum': 0.0, 'LOAN_RATE': 0.06074926678103038, 
'CREDIT_INCOME_RATIO': 2.007888888888889, 'EXT_SOURCE_SUM': 0.48536134023828964, 'EXT_SOURCE_MEAN': 0.16178711341276322}
)

response = requests.request("POST", url, data=payload)

print(response.text)