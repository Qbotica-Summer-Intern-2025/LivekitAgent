from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv(".env")

# Load Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
trunk_sid = os.getenv("SIP_OUTBOUND_TRUNK_ID")
sip_uri = os.getenv("LIVEKIT_SIP_URI")
client = Client(account_sid, auth_token)


# Create origination URI
origination_url = client.trunking.v1.trunks(trunk_sid).origination_urls.create(
    sip_url=sip_uri,
    weight=1,
    priority=1,
    enabled=True,
    friendly_name="LiveKit Outbound"
)

print(f"Created outbound route: {origination_url.sid}")
