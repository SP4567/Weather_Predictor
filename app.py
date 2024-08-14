import pickle
import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

preci = st.number_input('Enter the precipitation', format="%.2f")
maxtemp = st.number_input('Enter the maximum temperature', format="%.2f")
mintemp = st.number_input('Enter the minimum temperature', format="%.2f")
windsp = st.number_input('Enter the wind speed', format="%.2f")

if st.button('Predict'):
    weather_cl = [[preci, maxtemp, mintemp, windsp]]
    inp = Scaler.transform(weather_cl)
    
    try:
        res = model.predict(inp)
        class_x = np.argmax(res, axis=1)[0]
        
        weather_classes = {
            1: "Drizzle",
            2: "Rain",
            3: "Sun",
            4: "Snow",
            5: "Fog"
        }
        
        # Display the result
        st.header(weather_classes.get(class_x, "Unknown"))
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Exception details:", e)
