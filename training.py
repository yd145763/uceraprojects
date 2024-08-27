# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:11:30 2024

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import MinMaxScaler

import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation



#==============================training dataset=======================
Pitch = 0.8e-06, 1.0e-06, 1.4e-06, 1.6e-06
steps = 5

df_data_main_main = pd.DataFrame()

for p in Pitch:
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
    for x in range(df.shape[1]):
        e = df.iloc[:, x]
        df_data = pd.DataFrame()
        df_data['z'] = range(df.shape[0])
        df_data['x'] = x
        df_data['p'] = p
        df_data['e'] = e

        df_data_main = pd.concat([df_data_main, df_data], axis = 0)
    
    df_data_main_main = pd.concat([df_data_main_main, df_data_main], axis = 0)

dataset = np.array(df_data_main_main)
X_train = dataset[:, :-1]
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

y_train = dataset[:, -1]
y_train = y_train.reshape(-1,1)
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train)
y_train = y_train[:,0]
y_train = pd.DataFrame(y_train)

#==============================validation dataset=======================


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
for x in range(df.shape[1]):
    e = df.iloc[:, x]
    df_data = pd.DataFrame()
    df_data['z'] = range(df.shape[0])
    df_data['x'] = x
    df_data['p'] = p
    df_data['e'] = e

    df_data_main = pd.concat([df_data_main, df_data], axis = 0)


dataset = np.array(df_data_main)
X_val_denorm = dataset[:, :-1]
scaler_x = MinMaxScaler()
X_val = scaler_x.fit_transform(X_val_denorm)
X_val[:, 2] = (p - min(Pitch))/(max(Pitch) - min(Pitch))
X_val = pd.DataFrame(X_val)

y_val_denorm = dataset[:, -1]
y_val_denorm = y_val_denorm.reshape(-1,1)
scaler_y = MinMaxScaler()
y_val = scaler_y.fit_transform(y_val_denorm)
y_val = y_val[:,0]
y_val = pd.DataFrame(y_val)


dense_layers = [40,30,20,10]
layer_sizes = [100,80,50,40,30]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        model = Sequential()
        model.add(Dense(len(X_train.keys()),  input_shape=[len(X_train.keys())]))
        for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('elu'))
                #layer_size = int(round(layer_size*0.9, 0))
        model.add(Dense(y_train.shape[1]))
        
        
        # Compile the model
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        
        start_time = time.time()
        import tensorflow as tf
        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=30,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights that yield the best validation loss
        )
        
        history = model.fit(X_train, y_train, epochs=100,validation_data=(X_val, y_val), batch_size = 2500
                            ,callbacks=[early_stopping]
                            )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        y_pred = model.predict(X_val)
        y_pred = scaler_y.inverse_transform(y_pred)
        
        df_pred = np.hstack((X_val_denorm, y_pred))
        df_pred = pd.DataFrame(df_pred)
        df_pred.columns = ['z', 'x', 'p', 'e']
        df_pred['x'] = [int(i) for i in df_pred['x']]
        df_pred['z'] = [int(i) for i in df_pred['z']]
        
        df_pred_contour = pd.DataFrame()
        
        for i in range(df.shape[1]):
            df_e = df_pred[df_pred['x'] ==i]
            e = df_e['e']
            e = e.reset_index(drop = True)
            df_pred_contour[i] = e
        
        
        #set the range of x and z
        x = np.linspace(-45, 70, num=df_pred_contour.shape[1])
        z = np.linspace(16, 90, num=df_pred_contour.shape[0])
        
        #plotting the contour plot of the whole df, the whole picture of light coupled out from grating
        colorbarmax = df_pred_contour.max().max()
        colorbarmin = df_pred_contour.min().min()
        
        if colorbarmin <0:
            df_pred_contour = df_pred_contour + abs(colorbarmin)
        

        
        X,Z = np.meshgrid(x,z)
        #contour plot 
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Z,df_pred_contour, 200, zdir='z', offset=-100, cmap='jet')
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
        plt.title('layer size '+str(layer_size)+'_dense layer'+str(dense_layer))
        plt.savefig('/home/grouptan/Documents/yudian/ucera/'+'layer size '+str(layer_size)+'_dense layer'+str(dense_layer)+'.png')
        plt.show()
        plt.close()
        
        model.summary()
