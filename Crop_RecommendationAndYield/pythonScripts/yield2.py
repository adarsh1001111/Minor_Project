import joblib
import pandas as pd
import numpy as np


pipeline = joblib.load('crop_yield_model.pkl')

def get_user_input():
    print("Please enter the following details:")
    crop = input("Enter the crop (e.g., Wheat, Rice, Corn): ")
    temperature = float(input("Enter the average temperature (°C): "))
    rainfall = float(input("Enter the average rainfall (mm): "))
    humidity = float(input("Enter the humidity (%): "))
    ph = float(input("Enter the soil pH: "))
    
    
    user_input = pd.DataFrame({
        'Crop': [crop],
        'Temperature': [temperature],
        'Rainfall': [rainfall],
        'Humidity': [humidity],
        'Soil pH': [ph]
    })
    
    return user_input


user_data = get_user_input()


predicted_yield = pipeline.predict(user_data)

print(f"Predicted Crop Yield: {predicted_yield[0]} kg/ha")
