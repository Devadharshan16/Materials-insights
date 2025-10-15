# predict.py (Final Version - Reads from Local CSVs & accepts weights)

import pandas as pd
from datetime import timedelta
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

materials_df = pd.DataFrame()
prices_df = pd.DataFrame()
vendors_df = pd.DataFrame()

def standardize_df_columns(df, column_map):
    df.columns = df.columns.str.strip().str.lower()
    for target_name, possible_names in column_map.items():
        if target_name in df.columns:
            continue
        for name in possible_names:
            if name in df.columns:
                df.rename(columns={name: target_name}, inplace=True)
                logging.info(f"Standardized column '{name}' to '{target_name}'.")
                break
    return df

def initialize_data_from_csv():
    global materials_df, prices_df, vendors_df
    MATERIALS_COL_MAP = {
        "material_id": ["material", "id", "code"],
        "name": ["description", "material_name"]
    }
    PRICES_COL_MAP = {
        "material_id": ["material", "id", "code"],
        "date": ["date"], "price": ["price"], "vendor_id": ["vendor"]
    }
    VENDORS_COL_MAP = {
        "material_id": ["material_id"], "vendor_id": ["vendor_id"],
        "reliability_score": ["reliability_score"],
        "delivery_days": ["delivery_days", "avg_delivery_days"],
        "price_per_unit": ["price_per_unit"]
    }
    try:
        logging.info("Attempting to load data from local CSV files...")
        materials_df = standardize_df_columns(pd.read_csv("data/materials.csv"), MATERIALS_COL_MAP)
        prices_df = standardize_df_columns(pd.read_csv("data/material_prices.csv"), PRICES_COL_MAP)
        vendors_df = standardize_df_columns(pd.read_csv("data/vendors.csv"), VENDORS_COL_MAP)
        logging.info("SUCCESS: All CSV files loaded and standardized.")
    except Exception as e:
        logging.error(f"FATAL ERROR while loading CSVs: {e}")

def get_materials():
    if 'material_id' not in materials_df.columns:
        raise ValueError("Missing 'material_id' in materials data.")
    if 'name' not in materials_df.columns:
        materials_df['name'] = materials_df['material_id']
    return materials_df[['material_id', 'name']].drop_duplicates().to_dict('records')

def get_price_prediction(material_id):
    required_cols = ['material_id', 'date', 'price']
    if not all(col in prices_df.columns for col in required_cols):
        raise ValueError("Prices data missing required columns.")
    df = prices_df[prices_df['material_id'].astype(str).str.lower() == str(material_id).lower()].copy()
    if df.empty or len(df) < 2: return None, None
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["price", "date"], inplace=True)
    if len(df) < 2: return None, None
    df["day_number"] = (df["date"] - df["date"].min()).dt.days
    X = df["day_number"].values.reshape(-1, 1)
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    last_day = df["day_number"].max()
    future_days = np.array(range(last_day + 1, last_day + 8)).reshape(-1, 1)
    future_dates = [df["date"].max() + timedelta(days=i) for i in range(1, 8)]
    predicted_prices = model.predict(future_days)
    predictions = [{"date": str(d.date()), "predicted_price": round(p, 2)} for d, p in zip(future_dates, predicted_prices)]
    std_dev = np.std(y)
    predictions_with_ci = [{**p, "confidence_low": round(p["predicted_price"] - std_dev, 2), "confidence_high": round(p["predicted_price"] + std_dev, 2)} for p in predictions]
    historical = df.copy()
    historical['date'] = historical['date'].dt.strftime('%Y-%m-%d')
    return historical.to_dict('records'), predictions_with_ci

def get_vendor_recommendation(material_id, weights):
    required = ['material_id', 'vendor_id', 'reliability_score', 'delivery_days', 'price_per_unit']
    if not all(col in vendors_df.columns for col in required):
        raise ValueError("Vendors data missing required columns.")
    df = vendors_df[vendors_df['material_id'].astype(str).str.lower() == str(material_id).lower()].copy()
    if df.empty: return None, [], {}
    for col in ['reliability_score', 'delivery_days', 'price_per_unit']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: return None, [], {}
    df['reliability_norm'] = df['reliability_score'] / 5.0
    df['delivery_norm'] = 1 - (df['delivery_days'] / df['delivery_days'].max()) if df['delivery_days'].max() > 0 else 0
    df['price_norm'] = 1 - (df['price_per_unit'] / df['price_per_unit'].max()) if df['price_per_unit'].max() > 0 else 0
    df['final_score'] = (df['reliability_norm'] * weights['reliability']) + \
                      (df['delivery_norm'] * weights['delivery']) + \
                      (df['price_norm'] * weights['price'])
    best_vendor = df.loc[df['final_score'].idxmax()]
    breakdown = {
        "reliability": best_vendor['reliability_norm'] * weights['reliability'],
        "delivery": best_vendor['delivery_norm'] * weights['delivery'],
        "price": best_vendor['price_norm'] * weights['price']
    }
    return best_vendor.to_dict(), df.sort_values(by='final_score', ascending=False).to_dict('records'), breakdown

initialize_data_from_csv()
