import pickle
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load scaler and model
Scaler = pickle.load(open("scaler.pkl", "rb"))
model = load_model("Weather_Predictor.h5")

def return_prediction(ANN, Scaler, sample_json):
    preci = sample_json["precipitation"]
    max_temp = sample_json["temp_max"]
    min_temp = sample_json["temp_min"]
    wind_speed = sample_json["wind"]
    weather_class = [[preci, max_temp, min_temp, wind_speed]]
    weather_class = Scaler.transform(weather_class)
    predict_x = ANN.predict(weather_class)
    classes_ind = np.argmax(predict_x, axis=1)
    return classes_ind

# Streamlit UI
st.title('Weather Sense')
preci = st.number_input('Enter the precipitation')
maxtemp = st.number_input('Enter the maximum temperature')
mintemp = st.number_input('Enter the minimum temperature')
windsp = st.number_input('Enter the wind speed')

if st.button('Predict'):
    weather_cl = [[preci, maxtemp, mintemp, windsp]]
    inp = Scaler.transform(weather_cl)
    try:
        res = model.predict(inp)
        class_x = np.argmax(res, axis=1)[0]
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
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        raise
