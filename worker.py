from livekit.agents import cli, AgentSession, JobContext, WorkerOptions, Agent, RoomInputOptions
from livekit import rtc, agents
from livekit.plugins import openai, deepgram, silero, noise_cancellation
import pandas as pd
import os
import random
import logging
import json
import asyncio
from dotenv import load_dotenv

load_dotenv(".env")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LogisticsAgent")

AGENT_NAME = "outbound-agent"
BOT_NAME = "Qubi"
BOT_COMPANY = "Qbotica"

async def log_transcript(role: str, content: str):
    logger.info(f"[{role.upper()}]: {content}")

class OutboundAgent(Agent):
    def __init__(self, room: rtc.Room = None):
        super().__init__(
            instructions=(
                f"You are an outbound caller agent named {AGENT_NAME}. Your main task is to make outbound calls to trucking companies to send loads from logistics companies acting as brokers. "
                "You have access to shipment data via the 'get_quote' function from a CSV file. "
                "Speak like a human, using fillers and pause words like 'uhmm', 'like', 'I mean', 'you know', 'so', 'I think so', 'I guess' to sound natural. "
                "Your primary goal is to negotiate the best price for the load. Steps: "
                "1. Present the load details (origin, destination, quote) and ask if the quote is acceptable. "
                "2. If they agree, lock in the rate and confirm sending dates. "
                "3. If they request a higher price (e.g., quote is $2000, they ask $2200), negotiate to reduce the price by around $300 (e.g., to $1900). "
                "4. If negotiation fails and they insist on a higher price, use the 'send_slack_message' function to ask for manager approval for their price. "
                "5. Wait for a Slack response (e.g., 'approved' or 'denied'). If approved, confirm the price; if denied or no response within 30 seconds, inform the trucking company you'll need to check with a manager and use the 'transfer_call' function to transfer the call. "
                "6. If the trucking company is not interested, use 'send_slack_message' to request manager approval to send the load to another company, then use 'get_quote' to get a new quote and repeat the process."
            )
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.shipment_data = self.load_shipment_data()
        self.slack_response = None
        self.slack_response_event = asyncio.Event()
        self.room = room

    def load_shipment_data(self) -> dict:
        try:
            df = pd.read_csv("test_dataset.csv")
            if df.empty:
                logger.error("Shipment dataset is empty")
                return {}
            random_row = df.sample(n=1).iloc[0].to_dict()
            logger.info(f"Selected random shipment: {random_row}")
            return random_row
        except FileNotFoundError:
            logger.error("test_dataset.csv file not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading shipment data: {e}")
            return {}

    @agents.function_tool(name="send_slack_message", description="Send a message to a Slack channel via the LiveKit room.")
    async def send_slack_message(self, ctx: agents.RunContext, message: str) -> dict:
        try:
            if not self.room:
                logger.error("Room not initialized in OutboundAgent")
                return {"status": "failed", "response": "Room not initialized"}
            data = {"message": message}
            payload = json.dumps(data).encode('utf-8')
            await self.room.local_participant.publish_data(payload=payload)
            logger.info(f"Published Slack message to room: message={message}")
            self.slack_response = None
            self.slack_response_event.clear()
            return {"status": "success", "response": "Message sent to Slack agent."}
        except Exception as e:
            logger.error(f"Error publishing Slack message: {e}")
            return {"status": "failed", "response": f"Failed to send message: {str(e)}"}

    @agents.function_tool(name="get_quote", description="Get a new quote from the shipment dataset.")
    async def get_quote(self, ctx: agents.RunContext) -> dict:
        self.shipment_data = self.load_shipment_data()
        if not self.shipment_data:
            return {"status": "failed", "response": "No shipment data available."}
        return {
            "status": "success",
            "quote": self.shipment_data.get("quote"),
            "origin": self.shipment_data.get("origin"),
            "destination": self.shipment_data.get("destination")
        }

    @agents.function_tool(name="transfer_call", description="Transfer the call to a manager.")
    async def transfer_call(self, ctx: agents.RunContext, reason: str) -> dict:
        try:
            await log_transcript("assistant", f"One moment, I'm transferring you to my manager due to {reason}.")
            await ctx.room.disconnect()
            logger.info(f"Call transferred to manager: {reason}")
            return {"status": "success", "response": "Call transferred to manager."}
        except Exception as e:
            logger.error(f"Error transferring call: {e}")
            return {"status": "failed", "response": f"Failed to transfer call: {str(e)}"}

    def use_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(room=job_ctx.room.name)

async def entrypoint(ctx: JobContext):
    logger.info(f"Job started in room: {ctx.room.name}")
    await ctx.connect()

    def on_data_received(packet: rtc.DataPacket):
        try:
            payload = packet.data.decode('utf-8')
            logger.debug(f"Received data packet: {payload}")
            data = json.loads(payload)
            slack_message = data.get("slack_message")
            if slack_message:
                logger.info(f"Received Slack message: {slack_message}")
                agent.slack_response = slack_message.lower()
                agent.slack_response_event.set()
        except json.JSONDecodeError:
            logger.error(f"Failed to parse data packet: {payload}")
        except Exception as e:
            logger.error(f"Error processing data packet: {e}")

    session = AgentSession(
        stt=deepgram.STT(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=True,
            endpointing_ms=300
        ),
        llm=openai.LLM(model="gpt-4o", temperature=0.7),
        tts=deepgram.TTS(model="aura-2-andromeda-en"),
        vad=silero.VAD.load()
    )

    agent = OutboundAgent(room=ctx.room)
    ctx.room.on("data_received", on_data_received)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        )
    )
    shipment = agent.shipment_data
    if not shipment:
        welcome_message = (
            "Hello! This is Qubi from Qbotica logistics. "
            "I'm sorry, but I couldn't find any shipment details at the moment. "
            "Can we discuss potential opportunities anyway?"
        )
    else:
        welcome_message = (
            f"Hello! This is {BOT_NAME} from {BOT_COMPANY} logistics. "
            f"I have a load opportunity, you know, from {shipment.get('origin')} "
            f"to {shipment.get('destination')} with a quote of ${shipment.get('quote')}. "
            "Is this something you'd be interested in? Is the quote okay?"
        )
    await log_transcript("assistant", welcome_message)
    await session.generate_reply(instructions=welcome_message)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))