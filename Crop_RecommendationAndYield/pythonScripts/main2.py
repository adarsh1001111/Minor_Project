import numpy as np
import pandas as pd
import pickle
import requests
import joblib


model_path = r'/Users/adarshamit1001/MInor project/Crop_RecommendationAndYield/test_models/RandomForest.pkl'
with open(model_path, 'rb') as model_file:
    RF = pickle.load(model_file)


def weather_fetch(city_name):
    
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        main_data = data["main"]
        temperature = round(main_data["temp"] - 273.15, 2)  
        humidity = main_data["humidity"]
        return temperature, humidity
    else:
        print("City not found.")
        return None, None


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    if input_data is not None:  
        prediction = RF.predict(input_data)
        print(f"Recommended crop: {prediction[0]}")
        return prediction[0]
    return 0

def yield_predict(crop, temperature, rainfall, humidity, ph):
    pipeline = joblib.load('/Users/adarshamit1001/MInor project/Crop_RecommendationAndYield/test_models/crop_yield_model.pkl')
    user_data = pd.DataFrame({
        'Crop': [crop],
        'Temperature': [temperature],
        'Rainfall': [rainfall*10],
        'Humidity': [humidity],
        'Soil pH': [ph]
    })
    predicted_yield = pipeline.predict(user_data)
    return round(predicted_yield[0],2)
temp,hum=weather_fetch("delhi")
yield_predict("banana",temp,200,hum,5)

