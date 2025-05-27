# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from dotenv import load_dotenv 

load_dotenv() 

# Your Twilio Account SID and Auth Token.
account_sid = os.environ.get("TWILIO_ACCOUNT_SID") 
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")  
# account_sid = "ACf9a92c5ab21b1b564751b76b8189218a" 
# auth_token = "68ad1d0c3ba4f6031c5d1170cb5a1a63" 

print(f"DEBUG: Account SID loaded: '{account_sid}'") 
print(f"DEBUG: Auth Token loaded: '{auth_token}'")   


# Initialize the Twilio client with your credentials
client = Client(account_sid, auth_token)

# Define the URL for your TwiML Bin.
# This URL points to the TwiML instructions that Twilio will execute
# when the recipient answers the outbound call.
# This replaces "http://demo.twilio.com/docs/voice.xml" with your custom TwiML Bin.
twiml_bin_url = "https://handler.twilio.com/twiml/EH8b8471d71a68250b66d6d9c75025beae"

# Make the outbound call
call = client.calls.create(
    url=twiml_bin_url,
    to="+14706956023",
    from_="+18444236613"
)

# Print the Call SID (a unique identifier for this call)
# You can use this SID to check the call status and details in your Twilio Console.
print(f"Call initiated with SID: {call.sid}")