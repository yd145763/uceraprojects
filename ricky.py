# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:16:51 2024

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

Pitch = 0.8e-06, 1.0e-06, 1.4e-06, 1.6e-06

df_main_main = pd.DataFrame()

for p in Pitch:
    
    steps = 10
    
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
    
    
    X,Z = np.meshgrid(x,z)
    
    #contour plot 
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Z,df, 200, zdir='z', offset=-100, cmap='jet')
    clb=fig.colorbar(cp, ticks=(np.linspace(colorbarmin, colorbarmax, num = 6)).tolist())
    clb.ax.set_title('E-field (eV)', fontweight="bold")
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

    Z = range(df.shape[0])
    
    df_main = pd.DataFrame()
    
    for z in Z:
        
        e = df.iloc[z, :]
        
        data = pd.DataFrame()
        data['e'] = e
        data['z'] = z
        data['x'] = range(df.shape[1])
        data['p'] = p
        data['RSI'] = ta.rsi(data['e'], length = 40)
        data['EMAS'] = ta.ema(data['e'], length = 20)
        data['EMAM'] = ta.ema(data['e'], length = 30)
        data['EMAL'] = ta.ema(data['e'], length = 40)
        data['next1'] = data['e'].shift(-1)
        
        data.dropna(inplace = True)
        data.reset_index(inplace = True, drop = True)
        
        df_main = pd.concat([df_main, data], axis = 0)
    
    df_main_main = pd.concat([df_main_main, df_main], axis = 0)

df_data_main_main = df_main_main
    
dataset = np.array(df_data_main_main)
X_train = dataset[:, :8]
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

y_train = dataset[:, 8:]
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train)


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
    
    df_data.dropna(inplace = True)
    df_data.reset_index(inplace = True, drop = True)

    df_data_main = pd.concat([df_data_main, df_data], axis = 0)

dataset = np.array(df_data_main)
X_val_denorm = dataset[:, :8]
scaler_x = MinMaxScaler()
X_val = scaler_x.fit_transform(X_val_denorm)
X_val[:, 3] = (p - min(Pitch))/(max(Pitch) - min(Pitch))
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

y_val_denorm = dataset[:, 8:]
scaler_y = MinMaxScaler()
y_val = scaler_y.fit_transform(y_val_denorm)


#functions to define transformer model

from tensorflow import keras
from tensorflow.keras import layers

num_transformer_blocks=1
num_heads = 2
ff_dim = 2
head_size = 16

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(y_val.shape[1])(x)
    return keras.Model(inputs, outputs)

input_shape = X_train.shape[1:]

model = build_model(
    input_shape,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=10,
    #callbacks=[early_stopping_callback],
    #callbacks=[checkpoint]
)
