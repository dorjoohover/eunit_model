
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

def load_median_data():
    try:
        median_df = pd.read_excel("median_data.xlsx")
        mileage_to_price_change = dict(zip(median_df["Км-ийн өсөлт"].astype(str), median_df["Үнийн өөрчлөлтийн хувь"]))
        if "300000" not in mileage_to_price_change:
            mileage_to_price_change["300000"] = median_df["Үнийн өөрчлөлтийн хувь"].iloc[-1] if not median_df.empty else -0.045
        logger.info("Loaded median_data.xlsx successfully")
        return mileage_to_price_change
    except FileNotFoundError:
        logger.error("median_data.xlsx not found")
        raise
    except Exception as e:
        logger.error(f"Error loading median_data.xlsx: {e}")
        raise

condition_mapping = {
    "00 гүйлттэй": 1,
    "Дугаар авсан": 2,
    "Дугаар аваагүй": 3,
}

def load_and_preprocess_data():
    try:
        df = pd.read_excel("Vehicles_ML_last_v_2_5.xlsx")
        columns = [
            "id", "Price", "Brand", "Mark", "Manifactured year", "Imported year",
            "Motor range", "engine", "gearBox", "khurd", "host", "color",
            "interier", "condition", "Distance"
        ]
        df = df[columns].dropna()
        logger.info(f"Loaded dataset with {len(df)} rows")

        for col in ["Price", "Manifactured year", "Imported year"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()

        invalid_years = df[df["Imported year"] < df["Manifactured year"]]
        if not invalid_years.empty:
            logger.warning(f"Dropping {len(invalid_years)} rows with Imported year < Manifactured year")
            df = df[df["Imported year"] >= df["Manifactured year"]]

        categorical_cols = [
            "Brand", "Mark", "Motor range", "engine", "gearBox",
            "khurd", "host", "color", "interier"
        ]
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        mileage_to_price_change = load_median_data()
        df["Distance"] = df["Distance"].astype(str)

        def parse_distance(dist):
            try:
                if dist == "300000":
                    return 300000
                elif '-' in dist:
                    low, high = map(int, dist.split('-'))
                    return (low + high) / 2
                return float(dist)
            except (ValueError, TypeError):
                logger.warning(f"Invalid Distance value: {dist}. Setting to 0")
                return 0

        df["Distance_encoded"] = df["Distance"].apply(parse_distance)

        def map_distance_to_median_range(dist):
            if dist == "300000":
                return "300000"
            try:
                low, high = map(int, dist.split('-')) if '-' in dist else (int(dist), int(dist))
                for median_range in mileage_to_price_change.keys():
                    if median_range == "300000" and dist == "300000":
                        return median_range
                    if '-' in median_range:
                        m_low, m_high = map(int, median_range.split('-'))
                        if low >= m_low and high <= m_high:
                            return median_range
                return "0-5000"
            except (ValueError, TypeError):
                return "0-5000"

        df["Distance_mapped"] = df["Distance"].apply(map_distance_to_median_range)

        valid_mileages = set(mileage_to_price_change.keys())
        unmapped = df[~df["Distance_mapped"].isin(valid_mileages)]["Distance"].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped Distance categories: {unmapped}. Setting to '0-5000'")
            df["Distance_mapped"] = df["Distance_mapped"].apply(lambda x: '0-5000' if x not in valid_mileages else x)

        df["Price_adjusted"] = df.apply(
            lambda row: row["Price"] * (1 + mileage_to_price_change.get(row["Distance_mapped"], 0)),
            axis=1
        )

        df["Price_range"] = pd.qcut(df["Price_adjusted"], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        logger.info("Class distribution in Price_range:\n" + str(df["Price_range"].value_counts()))

        df["condition"] = df["condition"].astype(str)
        df["condition_encoded"] = df["condition"].map(condition_mapping).fillna(3)
        unmapped_conditions = df[df["condition_encoded"].isna()]["condition"].unique()
        if len(unmapped_conditions) > 0:
            logger.warning(f"Unmapped condition categories: {unmapped_conditions}. Set to 'Дугаар аваагүй'")

        df["Age"] = df["Imported year"] - df["Manifactured year"]

        df["Distance_Age_Interaction"] = df["Distance_encoded"] * df["Age"]

        def parse_motor_range(motor):
            try:
                if '-' in motor:
                    return float(motor.split('-')[0])
                elif motor == 'Цахилгаан':
                    return 0.0
                return float(motor)
            except ValueError:
                logger.warning(f"Invalid Motor range value: {motor}. Setting to 0")
                return 0.0

        df["Distance_Motor_Interaction"] = df["Distance_encoded"] * df["Motor range"].map(parse_motor_range)

        logger.info(f"Preprocessed dataset with {len(df)} rows")
        return df, mileage_to_price_change
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def train_model(df):
    try:
        X = df.drop(["id", "Price", "Price_range", "Distance", "condition", "Price_adjusted", "Distance_mapped"], axis=1, errors='ignore')
        y = df["Price"]
        categorical_cols = [
            "Brand", "Mark", "Motor range", "engine", "gearBox",
            "khurd", "host", "color", "interier"
        ]
        numeric_cols = ["Manifactured year", "Imported year", "Distance_encoded", "condition_encoded", "Age", "Distance_Age_Interaction", "Distance_Motor_Interaction"]

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ])

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        param_grid = {
            'regressor__n_estimators': [200],
            'regressor__max_depth': [None],
            'regressor__min_samples_split': [5]
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model Performance: MAE = {mae:.2f}, R2 = {r2:.2f}")

        feature_names = (best_model.named_steps['preprocessor']
                         .transformers_[0][1].get_feature_names_out(categorical_cols).tolist() +
                         numeric_cols)
        importances = best_model.named_steps['regressor'].feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        logger.info("Feature Importances:\n" + str({k: v for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}))

        joblib.dump(best_model, 'model.pkl')
        logger.info("Model saved to model.pkl")

        return best_model, X_test, y_test, X_test.index
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise

def predict_price(model, input_data, df, mileage_to_price_change):
    try:
        input_df = pd.DataFrame([input_data])
        required_cols = [
            "Brand", "Mark", "Manifactured year", "Imported year",
            "Motor range", "engine", "gearBox", "khurd", "host",
            "color", "interier", "condition", "Distance"
        ]
        for col in required_cols:
            if col not in input_df:
                raise ValueError(f"Missing input column: {col}")

        categorical_cols = [
            "Brand", "Mark", "Motor range", "engine", "gearBox",
            "khurd", "host", "color", "interier"
        ]
        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str)

        input_df["Manifactured year"] = pd.to_numeric(input_df["Manifactured year"], errors="coerce")
        input_df["Imported year"] = pd.to_numeric(input_df["Imported year"], errors="coerce")
        if input_df[["Manifactured year", "Imported year"]].isna().any().any():
            raise ValueError("Invalid Manifactured year or Imported year. Must be numeric.")

        def parse_distance(dist):
            try:
                if isinstance(dist, (int, float)):
                    return float(dist)
                if dist == "300000":
                    return 300000
                elif '-' in dist:
                    low, high = map(int, dist.split('-'))
                    return (low + high) / 2
                return float(dist)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid Distance format: {dist}")

        input_df["Distance_encoded"] = input_df["Distance"].apply(parse_distance)

        def map_distance_to_median_range(dist):
            valid_ranges = mileage_to_price_change.keys()
            if dist in valid_ranges:
                return dist
            logger.warning(f"Invalid Distance range: {dist}. Setting to '0-5000'")
            return "0-5000"

        input_df["Distance_mapped"] = input_df["Distance"].apply(map_distance_to_median_range)
        valid_mileages = list(mileage_to_price_change.keys())
        if not input_df["Distance_mapped"].isin(valid_mileages).all():
            raise ValueError(f"Invalid Distance mapping. Valid options: {valid_mileages[:10]}...")

        input_df["condition"] = input_df["condition"].astype(str)
        input_df["condition_encoded"] = input_df["condition"].map(condition_mapping).fillna(3)

        input_df["Age"] = input_df["Imported year"] - input_df["Manifactured year"]
        if (input_df["Imported year"] < input_df["Manifactured year"]).any():
            raise ValueError("Imported year must be >= Manifactured year")

        def parse_motor_range(motor):
            try:
                if '-' in motor:
                    return float(motor.split('-')[0])
                elif motor == 'Цахилгаан':
                    return 0.0
                return float(motor)
            except ValueError:
                logger.warning(f"Invalid Motor range: {motor}")
                return 0.0

        input_df["Distance_Age_Interaction"] = input_df["Distance_encoded"] * input_df["Age"]
        input_df["Distance_Motor_Interaction"] = input_df["Distance_encoded"] * input_df["Motor range"].map(parse_motor_range)

        input_df = input_df.drop(["Distance", "condition", "Distance_mapped"], axis=1)

        pred_price = model.predict(input_df)[0]

        pred_price_adjusted = pred_price * (1 + mileage_to_price_change.get(input_df["Distance_mapped"].iloc[0], 0))

        price_bins = df["Price_adjusted"].quantile([0, 0.25, 0.5, 0.75, 1]).values
        if pred_price_adjusted <= price_bins[1]:
            pred_range = "Low"
        elif pred_price_adjusted <= price_bins[2]:
            pred_range = "Medium"
        elif pred_price_adjusted <= price_bins[3]:
            pred_range = "High"
        else:
            pred_range = "Very High"

        prob = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else np.array([0.25] * 4)
        classes = ['Low', 'Medium', 'High', 'Very High']
        ci_lower = np.clip(prob - 1.96 * np.std(prob), 0, 1)
        ci_upper = np.clip(prob + 1.96 * np.std(prob), 0, 1)

        return pred_range, pred_price_adjusted, prob, (ci_lower, ci_upper)
    except Exception as e:
        logger.error(f"Error in predict_price: {e}")
        raise

def evaluate_model(model, X_test, y_test, df, test_indices):
    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model Performance: MAE = {mae:.2f}, R2 = {r2:.2f}")

        errors = pd.DataFrame({
            "id": df.loc[test_indices, "id"],
            "Actual": y_test,
            "Predicted": y_pred,
            "Distance": df.loc[test_indices, "Distance"],
            "Mark": df.loc[test_indices, "Mark"],
        })
        errors["Error"] = np.abs(errors["Actual"] - errors["Predicted"])
        return errors
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise

def visualize_distance_price_relationship(model, X_test, df, test_indices, mileage_to_price_change):
    try:
        y_pred = model.predict(X_test)
        viz_df = pd.DataFrame({
            'Distance_encoded': df.loc[test_indices, 'Distance_encoded'],
            'Distance_mapped': df.loc[test_indices, 'Distance_mapped'],
            'Actual_Price': df.loc[test_indices, 'Price'],
            'Predicted_Price': y_pred
        })

        viz_df['Predicted_Price_Adjusted'] = viz_df.apply(
            lambda row: row['Predicted_Price'] * (1 + mileage_to_price_change.get(row['Distance_mapped'], 0)),
            axis=1
        )

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=viz_df, x='Distance_encoded', y='Actual_Price', color='blue', label='Actual', alpha=0.5)
        sns.scatterplot(data=viz_df, x='Distance_encoded', y='Predicted_Price_Adjusted', color='red', label='Predicted', alpha=0.5)
        plt.title('Actual vs Predicted Prices by Distance')
        plt.xlabel('Distance (km)')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig('price_prediction_scatter.png', bbox_inches='tight')
        plt.close()
        logger.info("Saved price prediction scatter plot to price_prediction_scatter.png")
    except Exception as e:
        logger.error(f"Error in visualize_distance_price_relationship: {e}")
        raise

def predict_all_data(model, df, mileage_to_price_change):
    try:
        if 'Distance_mapped' not in df.columns:
            def map_distance_to_median_range(dist):
                if dist == "300000":
                    return "300000"
                try:
                    low, high = map(int, dist.split('-')) if '-' in dist else (int(dist), int(dist))
                    for median_range in mileage_to_price_change.keys():
                        if median_range == "300000" and dist == "300000":
                            return median_range
                        if '-' in median_range:
                            m_low, m_high = map(int, median_range.split('-'))
                            if low >= m_low and high <= m_high:
                                return median_range
                    return "0-5000"
                except (ValueError, TypeError):
                    return "0-5000"

            df["Distance_mapped"] = df["Distance"].apply(map_distance_to_median_range)

        X_all = df.drop(["id", "Price", "Price_range", "Distance", "condition", "Price_adjusted", "Distance_mapped"], axis=1, errors='ignore')
        y_pred = model.predict(X_all)

        result_df = df[['id', 'Brand', 'Mark', 'Manifactured year', 'Imported year', 'Motor range',
                        'engine', 'gearBox', 'khurd', 'host', 'color', 'interier', 'condition',
                        'Distance', 'Price', 'Price_adjusted', 'Price_range', 'Distance_mapped']].copy()
        result_df['Predicted_Price'] = y_pred
        result_df['Predicted_Price_Adjusted'] = result_df.apply(
            lambda row: row['Predicted_Price'] * (1 + mileage_to_price_change.get(row['Distance_mapped'], 0)),
            axis=1
        )

        price_bins = df["Price_adjusted"].quantile([0, 0.25, 0.5, 0.75, 1]).values
        def map_to_range(price):
            if price <= price_bins[1]:
                return "Low"
            elif price <= price_bins[2]:
                return "Medium"
            elif price <= price_bins[3]:
                return "High"
            else:
                return "Very High"

        result_df['Predicted_Price_Range'] = result_df['Predicted_Price_Adjusted'].apply(map_to_range)

        for class_name in ['Low', 'Medium', 'High', 'Very High']:
            result_df[f'Prob_{class_name}'] = 0.25
            result_df[f'CI_Lower_{class_name}'] = 0.2
            result_df[f'CI_Upper_{class_name}'] = 0.3

        result_df.to_excel('predicted_prices_all_data.xlsx', index=False)
        logger.info("Saved predictions to predicted_prices_all_data.xlsx")
    except Exception as e:
        logger.error(f"Error in predict_all_data: {e}")
        raise

if __name__ == "__main__":
    try:
        df, mileage_to_price_change = load_and_preprocess_data()
        model, X_test, y_test, test_indices = train_model(df)
        errors = evaluate_model(model, X_test, y_test, df, test_indices)
        visualize_distance_price_relationship(model, X_test, df, test_indices, mileage_to_price_change)
        predict_all_data(model, df, mileage_to_price_change)

        example_input = {
            "Brand": "Toyota",
            "Mark": "Corolla",
            "Manifactured year": 2018,
            "Imported year": 2019,
            "Motor range": "1.6-2.0",
            "engine": "Бензин",
            "gearBox": "Автомат",
            "khurd": "Буруу",
            "host": "Урд",
            "color": "Цагаан",
            "interier": "Хар",
            "condition": "Дугаар авсан",
            "Distance": "10000-15000",
        }
        pred_range, pred_price, prob, (ci_lower, ci_upper) = predict_price(model, example_input, df, mileage_to_price_change)
        logger.info(f"Predicted Price Range for Distance 10000-15000: {pred_range}")
        logger.info(f"Predicted Price: {pred_price:.2f}")
        logger.info(f"Probabilities: {dict(zip(['Low', 'Medium', 'High', 'Very High'], prob))}")
        logger.info(f"Confidence Interval: {ci_lower}, {ci_upper}")

        distances = ["0-5000", "5000-10000", "10000-15000", "100000-105000", "295000-300000"]
        for dist in distances:
            example_input["Distance"] = dist
            pred_range, pred_price, prob, (ci_lower, ci_upper) = predict_price(model, example_input, df, mileage_to_price_change)
            logger.info(f"Distance: {dist}, Predicted Price Range: {pred_range}, Predicted Price: {pred_price:.2f}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")