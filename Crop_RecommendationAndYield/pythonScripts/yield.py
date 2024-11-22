import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('synthetic_crop_yield_data.csv')


X = df.drop('Yield', axis=1)  
y = df['Yield']               


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Crop'])
    ],
    remainder='passthrough'
)


model = RandomForestRegressor(n_estimators=100, random_state=42)


pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"average yield : {y.mean()}")

import joblib


joblib.dump(pipeline, 'crop_yield_model.pkl')
print("Model saved as crop_yield_model.pkl")
