import keras as k  #machine learning library
from keras.models import  Sequential, load_model  #model- Sequential
from keras.layers import Dense   #neural network dense layer(Each neuron recieves input from all the neurons in the previous layer)
import numpy as np  #num py pythton library use for numerical works
import pandas as pd  #pandas library use data manipulation and analysis
from sklearn.model_selection import train_test_split   #machine learining library
from sklearn.preprocessing import LabelEncoder,MinMaxScaler  #for lableEncoder use for converting strig in to number
import matplotlib.pyplot as plt  #use for plot our data
import os
import flask
import pickle

columns_to_retain=['al','hemo','pcc','rbcc','age','bp','bu','sod','pot','appet','dm','bgr','classification']  #select columns
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)  # filter selected columns
df = df.dropna(axis=0) # drop missing values or N/A data row