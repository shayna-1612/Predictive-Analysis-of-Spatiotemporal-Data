import pandas as pd
import matplotlib.pyplot as plt 

#df = pd.read_csv('C:/sem 7/CS SOP Mukesh Rohil/hungary_chickenpox.csv',index_col='Date',parse_dates=True)
df = pd.read_csv('C:/sem 7/CS SOP Mukesh Rohil/hungary_chickenpox.csv')
# print(df['Date'])
# df.index = pd.DatetimeIndex(df.index).to_period('W')
print(df.index)
df=df.dropna()
print('Shape of data:',df.shape)
df.head()
df
df['BARANYA'].plot(figsize=(15,5))
#plt.show()

# from pmdarima import auto_arima
# stepwise_fit = auto_arima(df['BUDAPEST'], trace=True,suppress_warnings=True) #best model arima (2,0,3)
# stepwise_fit = auto_arima(df['BARANYA'], trace=True,suppress_warnings=True) #best model arima (4,0,2)
# stepwise_fit = auto_arima(df['BACS'], trace=True,suppress_warnings=True) #best model arima (2,0,2)
# stepwise_fit = auto_arima(df['BEKES'], trace=True,suppress_warnings=True) #best model arima (0,1,1)
# stepwise_fit = auto_arima(df['BORSOD'], trace=True,suppress_warnings=True) #best model arima (2,0,3)
# stepwise_fit = auto_arima(df['CSONGRAD'], trace=True,suppress_warnings=True) #best model arima (1,0,1)
# stepwise_fit = auto_arima(df['FEJER'], trace=True,suppress_warnings=True) #best model arima (1,0,3)
# stepwise_fit = auto_arima(df['GYOR'], trace=True,suppress_warnings=True) #best model arima (3,0,2)
# stepwise_fit = auto_arima(df['HAJDU'], trace=True,suppress_warnings=True) #best model arima (3,0,2)
# stepwise_fit = auto_arima(df['HEVES'], trace=True,suppress_warnings=True) #best model arima (1,0,1)
# stepwise_fit = auto_arima(df['JASZ'], trace=True,suppress_warnings=True) #best model arima (3,0,1)
# stepwise_fit = auto_arima(df['KOMAROM'], trace=True,suppress_warnings=True) #best model arima (1,0,1)
# stepwise_fit = auto_arima(df['NOGRAD'], trace=True,suppress_warnings=True) #best model arima (2,0,1)
# stepwise_fit = auto_arima(df['PEST'], trace=True,suppress_warnings=True) #best model arima (4,0,2)
# stepwise_fit = auto_arima(df['SOMOGY'], trace=True,suppress_warnings=True) #best model arima (2,0,3)
# stepwise_fit = auto_arima(df['SZABOLCS'], trace=True,suppress_warnings=True) #best model arima (2,1,1)
# stepwise_fit = auto_arima(df['TOLNA'], trace=True,suppress_warnings=True) #best model arima (3,1,2)
# stepwise_fit = auto_arima(df['VAS'], trace=True,suppress_warnings=True) #best model arima (0,1,1)
# stepwise_fit = auto_arima(df['VESZPREM'], trace=True,suppress_warnings=True) #best model arima (2,3,1)
# stepwise_fit = auto_arima(df['ZALA'], trace=True,suppress_warnings=True) #best model arima (2,0,1)

train=df.iloc[:-30]
test=df.iloc[-30:]
print('Shape of training data:',train.shape)
print('Shape of testing data:', test.shape)

from statsmodels.tsa.arima.model import ARIMA
model1=ARIMA(train['BUDAPEST'],order=(2,0,3))
model1=model1.fit(method_kwargs={'maxiter':300})
model1.summary()

model2=ARIMA(train['BARANYA'],order=(4,0,2))
model2=model2.fit(method_kwargs={'maxiter':300})
model2.summary()

# model3=ARIMA(train['BACS'],order=(2,0,2))
# model3=model3.fit(method_kwargs={'maxiter':300})
# model3.summary()

# model4=ARIMA(train['BEKES'],order=(1,1,1))
# model4=model4.fit(method_kwargs={'maxiter':300})
# model4.summary()

# model5=ARIMA(train['BORSOD'],order=(2,0,3))
# model5=model5.fit(method_kwargs={'maxiter':300})
# model5.summary()

# model6=ARIMA(train['CSONGRAD'],order=(1,0,1))
# model6=model6.fit(method_kwargs={'maxiter':300})
# model6.summary()

# model7=ARIMA(train['FEJER'],order=(1,0,3))
# model7=model7.fit(method_kwargs={'maxiter':300})
# model7.summary()

# model8=ARIMA(train['GYOR'],order=(3,0,2))
# model8=model8.fit(method_kwargs={'maxiter':300})
# model8.summary()

# model9=ARIMA(train['HAJDU'],order=(3,0,2))
# model9=model9.fit(method_kwargs={'maxiter':300})
# model9.summary()

# model10=ARIMA(train['HEVES'],order=(1,0,1))
# model10=model10.fit(method_kwargs={'maxiter':300})
# model10.summary()

# model11=ARIMA(train['JASZ'],order=(3,0,1))
# model11=model11.fit(method_kwargs={'maxiter':300})
# model11.summary()

# model12=ARIMA(train['KOMAROM'],order=(1,0,1))
# model12=model12.fit(method_kwargs={'maxiter':300})
# model12.summary()

# model13=ARIMA(train['NOGRAD'],order=(2,0,1))
# model13=model13.fit(method_kwargs={'maxiter':300})
# model13.summary()

# model14=ARIMA(train['PEST'],order=(4,0,2))
# model14=model14.fit(method_kwargs={'maxiter':300})
# model14.summary()

# model15=ARIMA(train['SOMOGY'],order=(2,0,3))
# model15=model15.fit(method_kwargs={'maxiter':300})
# model15.summary()

# model16=ARIMA(train['SZABOLCS'],order=(2,1,1))
# model16=model16.fit(method_kwargs={'maxiter':300})
# model16.summary()

# model17=ARIMA(train['TOLNA'],order=(3,1,2))
# model17=model17.fit(method_kwargs={'maxiter':300})
# model17.summary()

# model18=ARIMA(train['VAS'],order=(1,1,1))
# model18=model18.fit(method_kwargs={'maxiter':300})
# model18.summary()

# model19=ARIMA(train['VESZPREM'],order=(2,3,1))
# model19=model19.fit(method_kwargs={'maxiter':300})
# model19.summary()

# model20=ARIMA(train['ZALA'],order=(2,0,1))
# model20=model20.fit(method_kwargs={'maxiter':300})
# model20.summary()

#to predict
start=len(train)
end=len(train)+len(test)-1
pred=model2.predict(start=start,end=end).rename('ARIMA Predictions')
pred.plot(legend=True)
test['BARANYA'].plot(legend=True)
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
test['BUDAPEST'].mean()
rmse=sqrt(mean_squared_error(pred,test['BUDAPEST']))
print('rmse:', rmse)

