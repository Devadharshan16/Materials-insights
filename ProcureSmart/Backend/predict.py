import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- GLOBAL DATA STORES & STATUS FLAG ---
materials_df = pd.DataFrame()
prices_df = pd.DataFrame()
vendors_df = pd.DataFrame()
initialization_success = False # This flag will track if data loading was successful
SYSTEM_REMINDERS = [] # NEW: In-memory store for reminders

def standardize_df_columns(df, column_map):
    df.columns = df.columns.str.strip().str.lower()
    for target_name, possible_names in column_map.items():
        if target_name in df.columns: continue
        for name in possible_names:
            if name in df.columns:
                df.rename(columns={name: target_name}, inplace=True)
                logging.info(f"Standardized column '{name}' to '{target_name}'.")
                break
    return df

def initialize_data_from_csv():
    """Loads and standardizes data from CSV files and sets the success flag."""
    global materials_df, prices_df, vendors_df, initialization_success
    
    MATERIALS_COL_MAP = {"material_id": ["material", "id"], "name": ["description", "material_name"]}
    PRICES_COL_MAP = {"material_id": ["material", "id"], "date": ["day"], "price": ["cost"], "vendor_id": ["vendor"]}
    VENDORS_COL_MAP = {"material_id": ["material_id", "material"], "vendor_id": ["vendor_id", "vendor"], "reliability_score": ["reliability"], "delivery_days": ["avg_delivery_days"], "price_per_unit": ["price"]}
    
    try:
        logging.info("Loading data from local CSV files...")
        materials_df = standardize_df_columns(pd.read_csv("data/materials.csv"), MATERIALS_COL_MAP)
        prices_df = standardize_df_columns(pd.read_csv("data/material_prices.csv"), PRICES_COL_MAP)
        vendors_df = standardize_df_columns(pd.read_csv("data/vendors.csv"), VENDORS_COL_MAP)
        if 'name' not in materials_df.columns: materials_df['name'] = materials_df['material_id']
        logging.info("SUCCESS: All CSV files loaded and standardized.")
        initialization_success = True # Set flag to True on successful load
    except Exception as e:
        logging.error(f"FATAL ERROR while loading CSV data: {e}")
        logging.error("Please ensure the 'data' folder exists inside 'Backend' and contains all three required CSV files with the correct column headers.")
        initialization_success = False # Set flag to False on failure

def get_materials():
    if 'material_id' not in materials_df.columns: raise ValueError("Missing 'material_id' in materials data.")
    return materials_df[['material_id', 'name']].drop_duplicates().to_dict('records')

def get_price_prediction(material_id):
    required = ['material_id', 'date', 'price']
    if not all(col in prices_df.columns for col in required): raise ValueError("Prices data missing required columns.")
    df = prices_df[prices_df['material_id'].astype(str).str.lower() == str(material_id).lower()].copy()
    if len(df) < 2: return None, None
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["price", "date"], inplace=True)
    if len(df) < 2: return None, None
    df["day_num"] = (df["date"] - df["date"].min()).dt.days
    X = df["day_num"].values.reshape(-1, 1)
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    last_day = df["day_num"].max()
    future_days = np.array(range(last_day + 1, last_day + 8)).reshape(-1, 1)
    future_dates = [df["date"].max() + timedelta(days=i) for i in range(1, 8)]
    predicted = model.predict(future_days)
    preds = [{"date": str(d.date()), "predicted_price": round(p, 2)} for d, p in zip(future_dates, predicted)]
    std_dev = np.std(y)
    preds_ci = [{**p, "confidence_low": round(p["predicted_price"] - std_dev, 2), "confidence_high": round(p["predicted_price"] + std_dev, 2)} for p in preds]
    hist = df.copy(); hist['date'] = hist['date'].dt.strftime('%Y-%m-%d')
    return hist.to_dict('records'), preds_ci

def get_vendor_recommendation(material_id, weights):
    required = ['material_id', 'vendor_id', 'reliability_score', 'delivery_days', 'price_per_unit']
    if not all(col in vendors_df.columns for col in required): raise ValueError("Vendors data missing required columns.")
    df = vendors_df[vendors_df['material_id'].astype(str).str.lower() == str(material_id).lower()].copy()
    if df.empty: return None, [], {}
    for col in ['reliability_score', 'delivery_days', 'price_per_unit']: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: return None, [], {}
    df['rel_norm'] = df['reliability_score'] / 5.0
    df['del_norm'] = 1 - (df['delivery_days'] / df['delivery_days'].max()) if df['delivery_days'].max() > 0 else 0
    df['price_norm'] = 1 - (df['price_per_unit'] / df['price_per_unit'].max()) if df['price_per_unit'].max() > 0 else 0
    df['final_score'] = (df['rel_norm'] * weights['reliability']) + (df['del_norm'] * weights['delivery']) + (df['price_norm'] * weights['price'])
    best = df.loc[df['final_score'].idxmax()]
    breakdown = {"reliability": best['rel_norm'] * weights['reliability'], "delivery": best['del_norm'] * weights['delivery'], "price": best['price_norm'] * weights['price']}
    
    # FIX: Manually convert numpy types to native Python types for JSON serialization
    best_vendor_dict = {
        'material_id': str(best['material_id']),
        'vendor_id': str(best['vendor_id']),
        'reliability_score': float(best['reliability_score']),
        'delivery_days': int(best['delivery_days']),
        'price_per_unit': float(best['price_per_unit']),
        'final_score': float(best['final_score'])
    }
    
    return best_vendor_dict, df.sort_values(by='final_score', ascending=False).to_dict('records'), breakdown

def check_requirement(requirement):
    material_id = requirement['material_id']
    deadline_str = requirement['deadline']
    quantity_needed = int(requirement['quantity'])
    days_to_deadline = (datetime.strptime(deadline_str, '%Y-%m-%d').date() - datetime.now().date()).days

    if days_to_deadline < 0:
        message = "The requirement deadline cannot be in the past."
        details = f"You selected {deadline_str}, which has already passed."
        return {"status": "error", "title": "Invalid Deadline", "message": message, "details": details, "text_message": f"{message} {details}"}
    
    vendors = vendors_df[vendors_df['material_id'].astype(str).str.lower() == str(material_id).lower()].copy()
    if vendors.empty:
        message = f"There are no registered vendors for material {material_id}."
        details = "Please check your data or add vendors for this material."
        return {"status": "warning", "title": "No Suppliers Found", "message": message, "details": details, "text_message": f"{message} {details}"}
    
    feasible = vendors[vendors['delivery_days'] <= days_to_deadline]
    if feasible.empty:
        fastest = vendors.loc[vendors['delivery_days'].idxmin()]
        total_price = quantity_needed * fastest['price_per_unit']
        message = f"No vendor can deliver within your {days_to_deadline}-day deadline."
        details = f"The fastest option is {fastest['vendor_id']} ({int(fastest['delivery_days'])} days). Est. cost: ₹{total_price:,.2f}."
        vendor_details = {
            'vendor_id': str(fastest['vendor_id']), 
            'material_id': material_id, 
            'quantity': quantity_needed,
            'price_per_unit': f"₹{fastest['price_per_unit']:.2f}", 
            'total_price': f"₹{total_price:,.2f}",
            'delivery_days': int(fastest['delivery_days']), 
            'is_feasible': False
        }
        return {"status": "warning", "title": "Deadline Unmet", "message": message, "details": details, "text_message": f"{message} {details}", "vendor_details": vendor_details}
    
    best = feasible.loc[feasible['reliability_score'].idxmax()]
    total_price = quantity_needed * best['price_per_unit']
    message = f"Vendor '{best['vendor_id']}' can meet your requirement."
    details = f"Recommended based on reliability. Est. cost: ₹{total_price:,.2f} for {quantity_needed} units."
    
    # NEW: Set a reminder for 2 days before the deadline
    deadline_date = datetime.strptime(deadline_str, '%Y-%m-%d').date()
    reminder_date = deadline_date - timedelta(days=2)
    
    if reminder_date >= datetime.now().date():
        reminder = {
            "id": f"rem_{int(datetime.now().timestamp())}",
            "material_id": material_id,
            "quantity": quantity_needed,
            "deadline": deadline_str,
            "reminder_date": reminder_date.strftime('%Y-%m-%d'),
            "assigned_vendor": str(best['vendor_id']),
        }
        SYSTEM_REMINDERS.append(reminder)
        logging.info(f"Reminder set for {material_id} on {reminder['reminder_date']}")

    vendor_details = {
        'vendor_id': str(best['vendor_id']), 
        'material_id': material_id, 
        'quantity': quantity_needed,
        'price_per_unit': f"₹{best['price_per_unit']:.2f}", 
        'total_price': f"₹{total_price:,.2f}",
        'delivery_days': int(best['delivery_days']), 
        'is_feasible': True
    }
    return {"status": "success", "title": "Requirement Feasible", "message": message, "details": details, "text_message": f"{message} {details}", "vendor_details": vendor_details}

# NEW: Function to get system-level alerts
def get_system_alerts():
    """
    Checks for any pending reminders and returns them as alerts.
    Removes the reminder once it has been sent.
    """
    today_str = datetime.now().date().strftime('%Y-%m-%d')
    upcoming_alerts = []
    
    # Iterate over a copy of the list to safely remove items
    for reminder in SYSTEM_REMINDERS[:]:
        if reminder["reminder_date"] <= today_str:
            alert_message = (
                f"Reminder: Order {reminder['quantity']} units of {reminder['material_id']} "
                f"from {reminder['assigned_vendor']}. Deadline is {reminder['deadline']}."
            )
            upcoming_alerts.append({
                "id": reminder["id"],
                "message": alert_message
            })
            SYSTEM_REMINDERS.remove(reminder) # Remove after processing
            logging.info(f"Triggered and removed reminder: {reminder['id']}")

    return upcoming_alerts

# This line runs when the file is first imported by app.py
initialize_data_from_csv()

