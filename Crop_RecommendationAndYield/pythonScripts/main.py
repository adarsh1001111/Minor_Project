import numpy as np
import pandas as pd
import pickle


model_path = r'D:\college\MInor project\RandomForest.pkl'
with open(model_path, 'rb') as model_file:
    RF = pickle.load(model_file)

def get_user_input():
    print("Please enter the following details for crop recommendation:")
    N = float(input("Nitrogen (N) content in soil: "))
    P = float(input("Phosphorus (P) content in soil: "))
    K = float(input("Potassium (K) content in soil: "))
    temperature = float(input("Temperature (in °C): "))
    humidity = float(input("Humidity (in %): "))
    ph = float(input("pH value of soil: "))
    rainfall = float(input("Rainfall (in mm): "))
    
    return pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

def recommend_crop():
    
    input_data = get_user_input()

    prediction = RF.predict(input_data)
    print(f"Recommended crop: {prediction[0]}")

if __name__ == "__main__":
    recommend_crop()
