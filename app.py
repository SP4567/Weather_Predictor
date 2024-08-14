import pickle
import streamlit as st
import keras
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.web.cli as stcli
from keras.src.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras.src.saving
def return_prediction(ANN, Scaler, sample_json):
    preci = sample_json["precipitation"]
    max_temp = sample_json["temp_max"]
    min_temp = sample_json["temp_min"]
    wind_speed = sample_json["wind"]
    weather_class = [[preci, max_temp, min_temp, wind_speed]]
    weather_class = Scaler.fit_transform(weather_class)
    predict_x = ANN.predict(weather_class)
    classes_ind = np.argmax(predict_x, axis=1)
    return classes_ind
Scaler = pickle.load(open("C:\\Users\\Suyash Pandey\\PycharmProjects\\Weather_Predictor\\scaler.pkl", "rb"))
model = load_model('Weather_Predictor.h5')
st.title('Weather_Sense')
Preci = st.number_input('Enter the precipitation')
maxtemp = st.number_input('Enter the maximum temperature')
mintemp = st.number_input('Enter the minimum temperature')
windsp = st.number_input('Enter the wind speed')
if st.button('Predict'):
    weather_cl = [[Preci, maxtemp, mintemp, windsp]]
    inp = Scaler.fit_transform(weather_cl)
    res = model.predict(inp)
    class_x = np.argmax(res, axis=1)
    if class_x == 1:
        st.header("Drizzle")
    elif class_x == 2:
        st.header("Rain")
    elif class_x == 3:
        st.header("Sun")
    elif class_x == 4:
        st.header("Snow")
    elif class_x == 5:
        st.header("Fog")
