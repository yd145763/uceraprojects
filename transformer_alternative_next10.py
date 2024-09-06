# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:08:58 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:17:36 2024

@author: limyu
"""


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_absolute_error
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from matplotlib.ticker import StrMethodFormatter
import pandas_ta as ta
model = tf.keras.models.load_model("C:\\Users\\limyu\\Google Drive\\ucera\\tf_model10")
Pitch = 0.8e-06, 1.0e-06, 1.4e-06, 1.6e-06

steps = 2
p = 1.2e-6
df = pd.read_csv('https://raw.githubusercontent.com/yd145763/uceraprojects/main/tl1e-05_p'+str(p)+'_wl1.092e-06_dc0.5_ed4e-07_.csv', index_col = 0)
df = df.iloc[100:, :]
df = df.iloc[::steps, ::steps]
df = df.reset_index(drop=True)
df.columns = pd.RangeIndex(start=0, stop=len(df.columns), step=1)


#set the range of x and z
x = np.linspace(-45, 70, num=df.shape[1])
z = np.linspace(16, 90, num=df.shape[0])

#plotting the contour plot of the whole df, the whole picture of light coupled out from grating
colorbarmax = df.max().max()
colorbarmin = df.min().min()

colorbartick = 9

X,Z = np.meshgrid(x,z)
df1 = df.to_numpy()

#contour plot 
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(X,Z,df, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(colorbarmin, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=18, fontweight="bold", labelpad=1)


ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#plt.axhline(y=12, color='white', linestyle='--')
plt.show()
plt.close()

df_data_main = pd.DataFrame()
for z in range(df.shape[0]):
    e = df.iloc[z, :]
    df_data = pd.DataFrame()
    df_data['e'] = e
    df_data['z'] = z
    df_data['x'] = range(df.shape[1])
    df_data['p'] = p        
    df_data['RSI'] = ta.rsi(df_data['e'], length = 40)
    df_data['EMAS'] = ta.ema(df_data['e'], length = 20)
    df_data['EMAM'] = ta.ema(df_data['e'], length = 30)
    df_data['EMAL'] = ta.ema(df_data['e'], length = 40)
    df_data['next1'] = df_data['e'].shift(-1)
    df_data['next2'] = df_data['e'].shift(-2)
    df_data['next3'] = df_data['e'].shift(-3)
    df_data['next4'] = df_data['e'].shift(-4)
    df_data['next5'] = df_data['e'].shift(-5)
    df_data['next6'] = df_data['e'].shift(-6)
    df_data['next7'] = df_data['e'].shift(-7)
    df_data['next8'] = df_data['e'].shift(-8)
    df_data['next9'] = df_data['e'].shift(-9)

    
    df_data.dropna(inplace = True)
    df_data.reset_index(inplace = True, drop = True)

    df_data_main = pd.concat([df_data_main, df_data], axis = 0)
    print(z, p)


dataset = np.array(df_data_main)
X_val_denorm = dataset[:, :8]
df_X_val_denorm = pd.DataFrame(X_val_denorm)
df_X_val_denorm.columns = df_data_main.columns[:8]

min0x, max0x = min(X_val_denorm[:,0]), max(X_val_denorm[:,0])
min1x, max1x = min(X_val_denorm[:,1]), max(X_val_denorm[:,1])
min2x, max2x = min(X_val_denorm[:,2]), max(X_val_denorm[:,2])
min3x, max3x = min(Pitch), max(Pitch)
min4x, max4x = min(X_val_denorm[:,4]), max(X_val_denorm[:,4])
min5x, max5x = min(X_val_denorm[:,5]), max(X_val_denorm[:,5])
min6x, max6x = min(X_val_denorm[:,6]), max(X_val_denorm[:,6])
min7x, max7x = min(X_val_denorm[:,7]), max(X_val_denorm[:,7])

norm_dict_x = {
    0: [min0x, max0x],
    1: [min1x, max1x],
    2: [min2x, max2x],
    3: [min3x, max3x],
    4: [min4x, max4x],
    5: [min5x, max5x],
    6: [min6x, max6x],
    7: [min7x, max7x]
}

X_val = np.empty((X_val_denorm.shape[0], X_val_denorm.shape[1]), dtype=float)
for i in range(X_val_denorm.shape[1]):
    min_value = norm_dict_x[i][0]
    max_value = norm_dict_x[i][1]
    X_val[:, i] = (X_val_denorm[:, i] - min_value)/(max_value - min_value)

df_X_val = pd.DataFrame(X_val)
df_X_val.columns = df_data_main.columns[:8]

X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

y_val_denorm = dataset[:, 8:]
df_y_val_denorm = pd.DataFrame(y_val_denorm)
df_y_val_denorm.columns = df_data_main.columns[8:]

min0y, max0y = min(y_val_denorm[:,0]), max(y_val_denorm[:,0])
min1y, max1y = min(y_val_denorm[:,1]), max(y_val_denorm[:,1])
min2y, max2y = min(y_val_denorm[:,2]), max(y_val_denorm[:,2])
min3y, max3y = min(y_val_denorm[:,3]), max(y_val_denorm[:,3])
min4y, max4y = min(y_val_denorm[:,4]), max(y_val_denorm[:,4])
min5y, max5y = min(y_val_denorm[:,5]), max(y_val_denorm[:,5])
min6y, max6y = min(y_val_denorm[:,6]), max(y_val_denorm[:,6])
min7y, max7y = min(y_val_denorm[:,7]), max(y_val_denorm[:,7])
min8y, max8y = min(y_val_denorm[:,8]), max(y_val_denorm[:,8])

norm_dict_y = {
    0: [min0y, max0y],
    1: [min1y, max1y],
    2: [min2y, max2y],
    3: [min3y, max3y],
    4: [min4y, max4y],
    5: [min5y, max5y],
    6: [min6y, max6y],
    7: [min7y, max7y],
    8: [min8y, max8y],
}


y_val = np.empty((y_val_denorm.shape[0], y_val_denorm.shape[1]), dtype=float)
for i in range(y_val_denorm.shape[1]):
    min_value = norm_dict_y[i][0]
    max_value = norm_dict_y[i][1]
    a = y_val_denorm[:, i]
    b = (a - min_value)/(max_value - min_value)
    y_val[:, i] = b

df_y_val = pd.DataFrame(y_val)
df_y_val.columns = df_data_main.columns[8:]

y_pred = model.predict(X_val)
df_y_pred = pd.DataFrame(y_pred)
df_y_pred.columns = df_data_main.columns[8:]

y_pred_denorm = np.empty((y_pred.shape[0], y_pred.shape[1]), dtype=float)
for i in range(y_pred.shape[1]):
    min_value = norm_dict_y[i][0]
    max_value = norm_dict_y[i][1]
    b = y_pred[:, i]
    a = (b*(max_value - min_value)) + min_value
    y_pred_denorm[:, i] = a



df_full_actual = np.hstack([X_val_denorm, y_val_denorm])
df_full_actual = pd.DataFrame(df_full_actual)
df_full_actual.columns = df_data.columns

df_full_pred = np.hstack([X_val_denorm, y_pred_denorm])
df_full_pred = pd.DataFrame(df_full_pred)
df_full_pred.columns = df_data.columns

df_filtered_actual = df_full_actual[df_full_actual['z'] ==100]
df_filtered_pred = df_full_pred[df_full_pred['z'] ==100]

plt.plot(df_filtered_actual['next9'])
plt.plot(df_filtered_pred['next9'])
plt.show()

df_alternate_contour = pd.DataFrame()
df_actual_contour = pd.DataFrame()

for z in range(df.shape[0]):

    e = df.iloc[z, :]
    df_data = pd.DataFrame()
    df_data['e'] = e
    df_data['z'] = z
    df_data['x'] = range(df.shape[1])
    df_data['p'] = p        
    df_data['RSI'] = ta.rsi(df_data['e'], length = 40)
    df_data['EMAS'] = ta.ema(df_data['e'], length = 20)
    df_data['EMAM'] = ta.ema(df_data['e'], length = 30)
    df_data['EMAL'] = ta.ema(df_data['e'], length = 40)
    df_data['next1'] = df_data['e'].shift(-1)
    df_data['next2'] = df_data['e'].shift(-2)
    df_data['next3'] = df_data['e'].shift(-3)
    df_data['next4'] = df_data['e'].shift(-4)
    df_data['next5'] = df_data['e'].shift(-5)
    df_data['next6'] = df_data['e'].shift(-6)
    df_data['next7'] = df_data['e'].shift(-7)
    df_data['next8'] = df_data['e'].shift(-8)
    df_data['next9'] = df_data['e'].shift(-9)
    
    df_data.reset_index(inplace = True, drop = True)
    
    df_alternate = df_data.iloc[:40, :8]
    
    X = np.arange(40, df_data.shape[0], 10)
    for x in X:
    
        X_val_single_denorm = df_data.iloc[x, :8]
        df_X_val_single_denorm = pd.DataFrame(X_val_single_denorm)
        df_X_val_single_denorm = df_X_val_single_denorm.transpose()
        
        df_alternate = pd.concat([df_alternate, df_X_val_single_denorm], axis = 0)
        df_alternate = df_alternate.reset_index(drop = True) 
        X_val_single = []
        for i in range(8):
            min_value = norm_dict_x[i][0]
            max_value = norm_dict_x[i][1]
            a = X_val_single_denorm[i]
            b = (a - min_value)/(max_value - min_value)
            X_val_single.append(b)
        X_val_single = np.array(X_val_single)
        X_val_single = X_val_single.reshape(1, X_val_single.shape[0],1)
        y_pred_single = model.predict(X_val_single)
        print(x)
        y_pred_single = y_pred_single.reshape(y_pred_single.shape[1],1)
        y_pred_single_denorm = []
        for i in range(9):
            min_value = norm_dict_y[i][0]
            max_value = norm_dict_y[i][1]
            b = y_pred_single[i, 0]
            a = (b*(max_value - min_value))+min_value
            y_pred_single_denorm.append(a)
            
        s = [y_pred_single_denorm[0], 240, x+1, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
    
        s = [y_pred_single_denorm[1], 240, x+2, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
        s = [y_pred_single_denorm[2], 240, x+3, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
        s = [y_pred_single_denorm[3], 240, x+4, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
        s = [y_pred_single_denorm[4], 240, x+5, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
        s = [y_pred_single_denorm[5], 240, x+6, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
        s = [y_pred_single_denorm[6], 240, x+7, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
    
        s = [y_pred_single_denorm[7], 240, x+8, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
        
    
        s = [y_pred_single_denorm[8], 240, x+9, p, np.nan, np.nan, np.nan, np.nan]
        s = pd.Series(s)
        s.index = X_val_single_denorm.index
        s = pd.DataFrame(s)
        s = s.transpose()
        df_alternate = pd.concat([df_alternate, s], axis=0)
        df_alternate = df_alternate.reset_index(drop=True) 
        RSI = ta.rsi(df_alternate['e'], length = 40)
        EMAS = ta.ema(df_alternate['e'], length = 20)
        EMAM = ta.ema(df_alternate['e'], length = 30)
        EMAL = ta.ema(df_alternate['e'], length = 40)
        
        df_alternate.loc[df_alternate.index[-1], 'RSI'] = np.array(RSI)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAS'] = np.array(EMAS)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAM'] = np.array(EMAM)[-1]
        df_alternate.loc[df_alternate.index[-1], 'EMAL'] = np.array(EMAL)[-1]
    
    
    plt.plot(df_alternate['e'][:df_data.shape[0]])
    plt.plot(df_data['e'])
    plt.show()
    df_alternate_contour[z] = df_alternate['e'][:df_data.shape[0]]
    df_actual_contour[z] = df_data['e']
    

