from hubspot import HubSpot
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv('.env')
access_token = os.getenv('HUBSPOT_ACCESS_TOKEN')
api_endpoint = os.getenv('HUBSPOT_API_ENDPOINT', 'https://api.hubapi.com/crm/v3/objects/contacts')

def search_contact_by_phone(phone_number: str) -> dict:
    """
    Search for a contact in HubSpot by phone number.
    Returns contact details if found, otherwise returns error.
    """
    normalized_phone = f"+1{phone_number}" if not phone_number.startswith('+') else phone_number
    url = f"{api_endpoint}/search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "filterGroups": [
            {
                "filters": [
                    {
                        "propertyName": "phone",
                        "operator": "EQ",
                        "value": normalized_phone
                    }
                ]
            }
        ],
        "properties": ["firstname", "lastname", "email", "phone"]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if data.get('total', 0) > 0 and 'results' in data:
            contact = data['results'][0]
            return {
                "status": "success",
                "contact": contact,
                "message": f"Welcome {contact['properties'].get('firstname', 'Valued')} "
                           f"{contact['properties'].get('lastname', 'Customer')}! "
                           "How can I assist you today?"
            }
        return {
            "status": "not_found",
            "contact": None,
            "message": "Your phone number does not match our records. Please verify your identity before proceeding."
        }

    except requests.exceptions.RequestException as e:
        print(f"Failed to search contact: {e}")
        return {
            "status": "error",
            "contact": None,
            "message": "Error verifying phone number, please try again."
        }

def custInfo():
    response = requests.get(api_endpoint, headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }, params={"properties": "firstname,lastname,email,phone"})

    if response.status_code == 200:
        contact_data = response.json()
        print("Contact Information:")
        if 'properties' in contact_data:
            for prop, value in contact_data['properties'].items():
                print(f"{prop}: {value}")
        elif 'results' in contact_data:
            for contact in contact_data['results']:
                print(f"Contact ID: {contact['id']}")
                for prop, value in contact['properties'].items():
                    print(f"  {prop}: {value}")
                print("-" * 20)
    else:
        print(f"Failed to retrieve contact information. Status Code: {response.status_code}")
        print(f"Error Message: {response.text}")

    return response.json()


