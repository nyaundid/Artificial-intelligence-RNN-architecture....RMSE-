
# coding: utf-8

# In[460]:


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout, Flatten , TimeDistributed, AveragePooling1D
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[461]:


dataset =('Documents\\TS_data_HW_18F12.csv')
df = pd.read_csv(dataset,header=None, na_values=['-1'], index_col=False)


print("Starting file:")
print(df[0:30])

print("Ending file:")
print(df[-30:])


# In[462]:



df_train = df
df_test = df.iloc[-30:] 


# In[463]:


df


# In[464]:



df_train = df.iloc[:2970]
df_test = df.iloc[-30:] 

spots_train = df_train[15].tolist()
spots_test = df_test[15].tolist()

print("Training set has {} observations.".format(len(spots_train)))
print("Test set has {} observations.".format(len(spots_test)))


# In[465]:


df_train.shape


# In[467]:


j = df_train[15].tolist()


# In[468]:


j


# In[469]:


j2 = sorted(i for i in j if i >= 100)


# In[470]:


j2


# In[471]:


k = df_test[15].tolist()


# In[472]:


k2 = sorted(i for i in k if i >= 100)


# In[473]:


import numpy as np

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE-1):
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)
    
    
SEQUENCE_SIZE = 30
x_train,y_train = to_sequences(SEQUENCE_SIZE,spots_train)


print("Shape of training set: {}".format(x_train.shape))


# In[474]:


x_train


# In[475]:




print("Training set has {} observations.".format(len(spots_train)))
print("Test set has {} observations.".format(len(spots_test)))


# In[476]:


import numpy as np

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE-1):
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)
    
    
SEQUENCE_SIZE = 1
x_test,y_test = to_sequences(SEQUENCE_SIZE,spots_test)



print("Shape of test set: {}".format(x_test.shape))


# In[477]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
import numpy as np

print('Build model...')
model = Sequential()
model.add(LSTM(64, dropout=0.0, recurrent_dropout=0.0,input_shape=(None, 1)))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
print('Train...')

model.fit(x_train,y_train,callbacks=[monitor],verbose=2,epochs=10)


# In[478]:


trainingbigger200power = sorted(i for i in spots_train if i >= 100)


# In[492]:


pred1 = model.predict(x_train)
score1 = np.sqrt(metrics.mean_squared_error(pred1,y_train))
print("Score (RMSE): {}".format(score1))


# In[480]:


trainingbigger200power


# In[493]:


from sklearn import metrics


pred2 = model.predict(x_test)
score2 = np.sqrt(metrics.mean_squared_error(pred2,y_test))
print("Score (RMSE): {}".format(score2))


# In[494]:


print("Training set has {} observations.".format(len(trainingbigger200power)))


# In[483]:


testbigger200power = sorted(i for i in spots_test  if i >= 100)


# In[484]:


plt.plot(df)
plt.show()


# In[485]:


testbigger200power


# In[486]:


plt.plot(pred)
plt.show()


# In[487]:


plt.plot(trainingbigger200power)


# In[488]:


plt.plot(pred2)
plt.show()


# In[489]:


plt.plot(testbigger200power)


# In[490]:



plt.plot(pred)
plt.plot(pred2)
plt.show()


# In[313]:


plt.plot(df)
plt.show()


# In[495]:


plt.plot(score1)


# In[497]:


rmseplot = np.sqrt(metrics.mean_squared_error(pred1,y_train))


# In[499]:


plt.plot(rmseplot)

