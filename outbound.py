import asyncio
import os
from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPParticipantInfo
from livekit.agents import AgentSession, Agent
from livekit import rtc, agents
from inbound import get_quote_from_dataset


load_dotenv(".env")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIP_OUTBOUND_TRUNK_ID=os.getenv("SIP_OUTBOUND_TRUNK_ID")
LOGISTICS_PHONE_NUMBER=os.getenv("LOGISTICS_PHONE_NUMBER")

agent_name="Qubi"

async def main():
    livekit_api = api.LiveKitAPI()

    request = CreateSIPParticipantRequest(
        sip_trunk_id=SIP_OUTBOUND_TRUNK_ID,
        sip_call_to=LOGISTICS_PHONE_NUMBER,
        room_name="my-room",
        participant_identity="sip-test",
        participant_name="Test Caller",
        krisp_enabled=True,
        wait_until_answered=True
    )

    try:
        participant = await livekit_api.sip.create_sip_participant(request)
        print(f"Successfully created {participant}")
    except Exception as e:
        print(f"Error creating SIP participant: {e}")
    finally:
        await asyncio.Event().wait()


asyncio.run(main())