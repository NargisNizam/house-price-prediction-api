# 🏠 House Price Prediction API

A Machine Learning project that predicts house prices using **Linear Regression** and provides predictions through a **FastAPI REST API**.

## 📌 Project Overview

This project uses the California Housing dataset to build a machine learning model for predicting house prices.

The project includes:

- Data cleaning
- Missing value handling
- Duplicate checking
- Categorical feature encoding
- Train-test splitting
- Linear Regression model training
- Model evaluation
- FastAPI REST API
- Interactive Swagger API documentation

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Pickle

## 📂 Project Structure

```text
house-price-prediction-api/
│
├── housing.csv
├── train.py
├── main.py
├── model.pkl
├── requirements.txt
├── README.md
│
└── template/
🧹 Data Preprocessing

The dataset contains 20,640 records.

Data preprocessing includes:

Handling missing values
Checking duplicate rows
Encoding the ocean_proximity categorical feature
Separating features and target
Splitting the dataset into training and testing data
Dataset Split
Training samples: 16,512
Testing samples: 4,128
Features: 12
Target: median_house_value
🤖 Machine Learning Model

The project uses:

Linear Regression

The model is trained using the processed training dataset and then evaluated on the testing dataset.

📊 Model Performance
Metric	Result
MAE	50,670.74
MSE	4,908,476,721.16
RMSE	70,060.52
R² Score	0.6254

The model achieved an R² Score of 0.6254 on the test dataset.

🚀 Running the Project
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/house-price-prediction-api.git
2. Open the project folder
cd house-price-prediction-api
3. Install dependencies
pip install -r requirements.txt
4. Train the model
python train.py

This will train the model and generate:

model.pkl
5. Start the FastAPI server
uvicorn main:app --reload

The API will run at:

http://127.0.0.1:8000
📖 API Documentation

FastAPI provides interactive Swagger documentation.

Open:

http://127.0.0.1:8000/docs

From there, you can test the prediction endpoint directly.

🔮 Prediction Endpoint
POST /predict

The API accepts house-related features and returns the predicted house price.

Example response:

{
  "predicted_house_price": 185432.75
}
🎯 Project Goal

The main goal of this project is to demonstrate how a Machine Learning model can be trained, evaluated, saved, and deployed as a REST API using FastAPI.

👩‍💻 Author

Nargis Nizam

AI & Data Science Student | Machine Learning | Python | FastAPI

⭐ Future Improvements
Try advanced regression models such as Random Forest and Gradient Boosting
Improve model performance
Add a user-friendly web interface
Deploy the API online
Add input validation and better error handling
