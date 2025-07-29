import pandas as pd
import os
import json
from typing import Optional, List, Dict
import random
import math

ORDERS_CSV_PATH = "orders.csv"


def create_new_order(
        origin_zip: str,
        dest_zip: str,
        weight_lbs: float,
        freight_class: int = 125,
        base_rate_per_mile_cwt: float = 0.10,
        fuel_surcharge_percent: float = 20,
        accessorial_charges: float = 35,
        driver_id: Optional[str] = None,
        vendor_id: Optional[str] = None,
        skid_details: Optional[List[Dict[str, float]]] = None,
        is_stackable: bool = True,
        is_urgent: bool = False,
        delivery_appointment: bool = False,
        origin_country: str = "US",
        dest_country: str = "US",
        Quote: str =None,
        skid_dims: str =None,

) -> dict:
    try:
        shipment_id = f"SHP{random.randint(10000, 99999)}"
        created_at = pd.Timestamp.now().isoformat()
        order_data = {
            "shipment_id": shipment_id,
            "origin_zip": origin_zip,
            "dest_zip": dest_zip,
            "weight_lbs": weight_lbs,
            "freight_class": freight_class,
            "base_rate_per_mile_cwt": base_rate_per_mile_cwt,
            "fuel_surcharge_percent": fuel_surcharge_percent,
            "accessorial_charges": accessorial_charges,
            "driver_id": driver_id if driver_id else None,
            "vendor_id": vendor_id if vendor_id else None,
            "skid_details": json.dumps(skid_details) if skid_details else None,
            "is_stackable": is_stackable,
            "is_urgent": is_urgent,
            "delivery_appointment": delivery_appointment,
            "origin_country": origin_country,
            "dest_country": dest_country,
            "status": "pending",
            "created_at": created_at,
            "last_updated_timestamp": created_at,
            "Quote": Quote,
            "skid_dims": skid_dims if skid_dims else "unknown",

        }

        orders_df = pd.DataFrame([order_data])
        if os.path.exists(ORDERS_CSV_PATH):
            existing_df = pd.read_csv(ORDERS_CSV_PATH)
            orders_df = pd.concat([existing_df, orders_df], ignore_index=True)

        orders_df.to_csv(ORDERS_CSV_PATH, index=False, na_rep="")
        return {
            "status": "success",
            "response": f"New order created with Shipment ID {shipment_id} from {origin_zip} ({origin_country}) to {dest_zip} ({dest_country}), Weight: {weight_lbs} lbs, Freight Class: {freight_class}. It will be processed soon."
        }
    except Exception as e:
        return {"status": "failed", "response": f"Error creating order: {str(e)}"}


def update_order(
        shipment_id: str,
        **kwargs
) -> dict:
    try:
        if not os.path.exists(ORDERS_CSV_PATH):
            return {"status": "failed", "response": f"Order with Shipment ID {shipment_id} not found."}

        orders_df = pd.read_csv(ORDERS_CSV_PATH)
        if shipment_id.upper() not in orders_df["shipment_id"].str.upper().values:
            return {"status": "failed", "response": f"Order with Shipment ID {shipment_id} not found."}

        idx = orders_df.index[orders_df["shipment_id"].str.upper() == shipment_id.upper()].tolist()[0]
        for key, value in kwargs.items():
            if key == "skid_details" and value is not None:
                orders_df.at[idx, key] = json.dumps(value)
            else:
                orders_df.at[idx, key] = value if value is not None else None
        orders_df.at[idx, "last_updated_timestamp"] = pd.Timestamp.now().isoformat()

        orders_df.to_csv(ORDERS_CSV_PATH, index=False, na_rep="")
        return {
            "status": "success",
            "response": f"Order with Shipment ID {shipment_id} updated successfully."
        }
    except Exception as e:
        return {"status": "failed", "response": f"Error updating order: {str(e)}"}