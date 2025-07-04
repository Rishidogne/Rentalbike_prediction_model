# 🚴 Bike Rental Demand Prediction (Streamlit + ML)

This project predicts hourly bike rental demand using a Random Forest Regressor trained on the UCI Bike Sharing Dataset. It includes a live Streamlit dashboard.

## 🔧 Features
- Predicts rental count based on weather, time, season, etc.
- Interactive Streamlit UI
- Trained RandomForestRegressor
- UCI dataset used for training

## 🛠️ How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## 📁 Project Structure
- `app.py`: Streamlit frontend
- `bike_prediction.py`: Model training
- `models/`: Saved model files
- `data/hour.csv`: Input dataset
