import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/car_model.pkl")
MEDIAN_PATH = os.getenv("MEDIAN_PATH", "assets/median_data.xlsx")
DATA_PATH = os.getenv("DATA_PATH", "assets/Vehicles_ML_last_v_2_5.xlsx")
PREDICT_PATH = os.getenv("PREDICT_PATH", "assets/predicted_prices_all_data.xlsx")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "assets/output_with_app_columns.json")
HOST       = os.getenv("HOST", "0.0.0.0")
PORT       = int(os.getenv("PORT", 80))

# simple API-key for requests
API_KEY    = os.getenv("API_KEY", "changeme123")


KEY_PATH = os.getenv("KEY_PATH", '')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')