import re
import requests

GOOGLE_MAPS_API_KEY = ''  # Replace with your real API key


def detect_country(postal_code: str) -> str:
    postal_code = postal_code.strip().upper()
    if re.match(r"^\d{5}(-\d{4})?$", postal_code):
        return "US"
    elif re.match(r"^[A-Z]\d[A-Z][ ]?\d[A-Z]\d$", postal_code):
        return "CA"
    return "UNKNOWN"


def format_postal_code(postal_code: str, country: str) -> str:
    postal_code = postal_code.strip().upper().replace(" ", "")
    if country == "CA" and len(postal_code) == 6:
        postal_code = f"{postal_code[:3]} {postal_code[3:]}"
    return f"{postal_code}, {country}"


def check_postal_code(postal_code: str):
    country = detect_country(postal_code)

    if country == "UNKNOWN":
        print("❌ Invalid postal code format. Please enter a valid US or Canadian code.")
        return

    formatted_address = format_postal_code(postal_code, country)
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": formatted_address,
        "key": GOOGLE_MAPS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") == "OK":
        result = data["results"][0]
        location = result["geometry"]["location"]
        full_address = result["formatted_address"]
        print(f"✅ Valid postal code ({country}).")
        print(f"📍 Address: {full_address}")
        print(f"🌎 Location: {location}")
    else:
        print(f"❌ API returned: {data.get('status')}")
        print(data)


# Example usage
if __name__ == "__main__":
    user_input = input("Enter a US or Canadian postal code: ").strip()
    check_postal_code(user_input)
