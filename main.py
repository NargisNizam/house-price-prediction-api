from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="House Price Prediction API")

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is Working!"
    }


@app.post("/predict")
def predict(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity_INLAND: bool,
    ocean_proximity_ISLAND: bool,
    ocean_proximity_NEAR_BAY: bool,
    ocean_proximity_NEAR_OCEAN: bool
):
    features = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_proximity_INLAND,
        ocean_proximity_ISLAND,
        ocean_proximity_NEAR_BAY,
        ocean_proximity_NEAR_OCEAN
    ]])

    prediction = model.predict(features)

    return {
        "predicted_house_price": round(float(prediction[0]), 2)
    }