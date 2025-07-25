import sys
import json
import joblib
import numpy as np
import pandas as pd
import logging
import sklearn.compose._column_transformer
from config import MODEL_PATH, MEDIAN_PATH, DATA_PATH

import warnings
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.simplefilter("ignore", InconsistentVersionWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

condition_mapping = {
    "00 гүйлттэй": 1,
    "Дугаар авсан": 2,
    "Дугаар аваагүй": 3,
}

_model = None
_median = None
_data = None

def load_median_data():
    global _median
    try:
        if(_median is None):
            _median = pd.read_excel(MEDIAN_PATH)
        
        
        mileage_to_price_change = dict(
            zip(
                _median["Км-ийн өсөлт"].astype(str),
                _median["Үнийн өөрчлөлтийн хувь"],
            )
        )
        if "300000" not in mileage_to_price_change:
            mileage_to_price_change["300000"] = (
                _median["Үнийн өөрчлөлтийн хувь"].iloc[-1]
                if not _median.empty else -0.045
            )
        logger.info("Loaded median_data.xlsx successfully")
        return mileage_to_price_change
    except FileNotFoundError:
        logger.error("median_data.xlsx not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading median_data.xlsx: {str(e)}")
        sys.exit(1)

def load_model():
    global _model

    if _model is None:
        try:
            # Patch for backward compatibility
            # import sklearn.compose._column_transformer
            class _RemainderColsList(list):
                pass
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

            _model = joblib.load(MODEL_PATH)
            logger.info("Loaded model successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            sys.exit(1)
    return _model

def load_price_bins():
    global _data
    try:
        if(_data is None):
            _data = pd.read_excel(DATA_PATH)
        df = _data
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])
        mileage_to_price_change = load_median_data()

        def map_distance_to_median_range(dist):
            if dist == "300000": return "300000"
            try:
                low, high = map(int, dist.split("-")) if "-" in dist else (int(dist), int(dist))
                for mr in mileage_to_price_change:
                    if mr == "300000" and dist == "300000": return mr
                    if "-" in mr:
                        m_low, m_high = map(int, mr.split("-"))
                        if low >= m_low and high <= m_high: return mr
                return "0-5000"
            except: return "0-5000"

        df["Distance_mapped"] = df["Distance"].apply(map_distance_to_median_range)
        df["Price_adjusted"] = df.apply(lambda row: row["Price"] * (1 + mileage_to_price_change.get(row["Distance_mapped"], 0)), axis=1)
        price_bins = df["Price_adjusted"].quantile([0, 0.25, 0.5, 0.75, 1]).values
        logger.info("Loaded price bins from dataset")
        return price_bins
    except Exception as e:
        logger.error(f"Error loading price bins: {str(e)}")
        sys.exit(1)

def parse_distance(dist):
    try:
        if isinstance(dist, (int, float)): return float(dist)
        if dist == "300000": return 300000
        elif "-" in dist:
            low, high = map(int, dist.split("-"))
            return (low + high) / 2
        return float(dist)
    except: return 0

def map_distance_to_median_range(dist, mileage_to_price_change):
    try:
        dist_str = str(dist)
        if dist_str in mileage_to_price_change: return dist_str
        if "-" in dist_str:
            low, high = map(int, dist_str.split("-"))
            for mr in mileage_to_price_change:
                if mr == "300000" and dist_str == "300000": return mr
                if "-" in mr:
                    m_low, m_high = map(int, mr.split("-"))
                    if low >= m_low and high <= m_high: return mr
        elif float(dist_str).is_integer():
            dist_num = float(dist_str)
            for mr in mileage_to_price_change:
                if mr == "300000" and dist_num >= 300000: return "300000"
                if "-" in mr:
                    m_low, m_high = map(int, mr.split("-"))
                    if m_low <= dist_num <= m_high: return mr
        return "0-5000"
    except: return "0-5000"

def parse_motor_range(motor):
    try:
        if motor == "Цахилгаан": return 0.0
        if "-" in motor: return float(motor.split("-")[0])
        return float(motor)
    except: return 0.0

def predict_price(input_data, model, mileage_to_price_change, price_bins):
    try:
        input_mapped = {
            "Brand": input_data["brand"],
            "Mark": input_data["mark"],
            "Manifactured year": input_data["Year_of_manufacture"],
            "Imported year": input_data["Year_of_entry"],
            "Motor range": input_data["Engine_capacity"],
            "engine": input_data["Engine"],
            "gearBox": input_data["Gearbox"],
            "khurd": input_data["Hurd"],
            "host": input_data["Drive"],
            "color": input_data["Color"],
            "interier": input_data["Interior_color"],
            "condition": input_data["Conditions"],
            "Distance": input_data["Mileage"],
        }
        df = pd.DataFrame([input_mapped])
        for col in ["Brand", "Mark", "Motor range", "engine", "gearBox", "khurd", "host", "color", "interier"]:
            df[col] = df[col].astype(str)

        df["Manifactured year"] = pd.to_numeric(df["Manifactured year"], errors="coerce")
        df["Imported year"] = pd.to_numeric(df["Imported year"], errors="coerce")
        df["Distance_encoded"] = df["Distance"].apply(parse_distance)
        df["Distance_mapped"] = df["Distance"].apply(lambda x: map_distance_to_median_range(x, mileage_to_price_change))
        df["condition"] = df["condition"].astype(str)
        df["condition_encoded"] = df["condition"].map(condition_mapping).fillna(3)
        df["Age"] = df["Imported year"] - df["Manifactured year"]
        df["Distance_Age_Interaction"] = df["Distance_encoded"] * df["Age"]
        df["Distance_Motor_Interaction"] = df["Distance_encoded"] * df["Motor range"].map(parse_motor_range)

        df.drop(["Distance", "condition", "Distance_mapped"], axis=1, inplace=True)

        pred_price = model.predict(df)[0]
        dist_mapped = map_distance_to_median_range(input_mapped["Distance"], mileage_to_price_change)
        pred_price_adjusted = pred_price * (1 + mileage_to_price_change.get(dist_mapped, 0))

        return {"predicted_price": float(pred_price_adjusted)}
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def predict(features) -> float:
    try:
        model = load_model()
        mileage_to_price_change = load_median_data()
        price_bins = load_price_bins()
        input_data = features

        if isinstance(input_data, list):
            results = [predict_price(d, model, mileage_to_price_change, price_bins) for d in input_data]
        else:
            results = predict_price(input_data, model, mileage_to_price_change, price_bins)
        print(json.dumps(results, ensure_ascii=False))
        return results
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)
