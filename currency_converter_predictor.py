import requests
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings

warnings.filterwarnings('ignore')

today = datetime.date(datetime.now())
sub_months = datetime.date(datetime.now()) + relativedelta(years=-15)
base = input("Convert from: ")
to = input("Convert to: ")
symbs = base.upper()+","+to.upper()
amount = float(input("Amount: "))

url = "https://api.exchangeratesapi.io/history?start_at=" +sub_months.strftime("%Y-%m-%d")+ "&end_at=" +today.strftime("%Y-%m-%d")+ "&base=" +base.upper()

response = requests.get(url) 
data = response.text
parsed = json.loads(data)  
rates = parsed["rates"]
df = pd.read_json(data)

complete_set={}
sub_set={}
complete_set = dict(rates.items())
for x,y in complete_set.items():
    for w,z in y.items():
        if(w == to.upper()):
            sub_set[x]=z

values_daily = pd.DataFrame(list((sub_set.items())), columns=['Date', 'Value'])
values_daily = values_daily.sort_values('Date')
values_daily['Date'] = pd.to_datetime(values_daily['Date'], format='%Y-%m-%d')
values_daily.set_index(['Date'], inplace=True)

X = values_daily.values
len_values = X.size
x_70 = math.ceil(0.7*len_values)
train = X[0:x_70]
test = X[x_70:]
predictions_ar = []
predictions_arima = []


##ARIMA model
p=d=q=range(0,5)
pdq = list(itertools.product(p,d,q))
model_arima_fit_aic_value_test = 1000
for param in pdq:
    try:
        model_arima = ARIMA(train, order=param)
        model_arima_fit = model_arima.fit()
        model_arima_fit_aic_value = model_arima_fit.aic
        if(model_arima_fit_aic_value<model_arima_fit_aic_value_test):
            model_arima_fit_aic_value_test = model_arima_fit_aic_value
            xparam = param
    except:
        continue

model_arima = ARIMA(train, order=xparam)
model_arima_fit = model_arima.fit()
predictions_arima = model_arima_fit.forecast(steps=len_values-x_70)[0]


##AR model
model_ar = AR(train)
model_ar_fit = model_ar.fit()
predictions_ar = model_ar_fit.predict(start=x_70, end=len_values-1)


mean_ar = mean_squared_error(test,predictions_ar)
mean_arima = mean_squared_error(test,predictions_arima)

if(mean_ar>mean_arima):
    predictions = model_arima_fit.forecast(steps=len_values)[0]
else:
    predictions = model_ar_fit.predict(start=x_70, end=len_values+x_70)
    
maximum_value = max(predictions)

print("Maximum Value: " +str(amount*maximum_value)+ "\n" + "Value of " +str(base.upper())+ " : " +str(maximum_value)+ "\n" +"After: "+str(np.argmax(predictions)+1)+" days")