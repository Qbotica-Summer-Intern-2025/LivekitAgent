import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("OrderManagementCleanup")

ORDERS_CSV_PATH = "orders.csv"
required_dtypes = {
    "shipment_id": str,
    "origin_zip": str,
    "dest_zip": str,
    "weight_lbs": float,
    "freight_class": float,
    "base_rate_per_mile_cwt": float,
    "fuel_surcharge_percent": float,
    "accessorial_charges": float,
    "driver_id": str,
    "vendor_id": str,
    "skid_details": str,
    "is_stackable": bool,
    "is_urgent": bool,
    "delivery_appointment": bool,
    "origin_country": str,
    "dest_country": str,
    "status": str,
    "created_at": str,
    "last_updated_timestamp": str,
    "skid_dims":str,
    "Quote":str
}

try:
    orders_df = pd.read_csv(ORDERS_CSV_PATH)
except Exception as e:
    logger.error(f"Failed to read orders.csv: {e}")
    orders_df = pd.DataFrame(columns=required_dtypes.keys())

for col in required_dtypes:
    if col not in orders_df.columns:
        orders_df[col] = None
for column, dtype in required_dtypes.items():
    if column in orders_df.columns:
        try:
            if dtype == str:
                orders_df[column] = orders_df[column].astype(str).replace({"nan": "", "NaN": "", "": "unknown"})
            elif dtype == float:
                orders_df[column] = pd.to_numeric(orders_df[column], errors="coerce")
            elif dtype == bool:
                orders_df[column] = orders_df[column].map({'True': True, 'False': False, True: True, False: False, 'nan': None})
        except Exception as e:
            logger.warning(f"Failed to cast {column} to {dtype}: {e}")

orders_df.to_csv(ORDERS_CSV_PATH, index=False, na_rep="")
logger.info("Cleaned and saved orders.csv")