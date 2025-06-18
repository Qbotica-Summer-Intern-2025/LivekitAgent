import asyncio
import logging
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Optional
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, RoomInputOptions
from livekit.plugins import deepgram, openai, silero, noise_cancellation
import random
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogisticsVoiceAgent")

AGENT_NAME = "outbound-logistics-agent"
LIVEKIT_ROOM_PREFIX = "outbound-call"

# Load environment variables
load_dotenv(".env.local")
required_env = [
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY", "OPENAI_API_KEY", "LOGISTICS_PHONE_NUMBER"
]
for var in required_env:
    if not os.getenv(var):
        raise ValueError(f"{var} missing in .env.local")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOGISTICS_PHONE_NUMBER = os.getenv("LOGISTICS_PHONE_NUMBER")

# Load logistics dataset
try:
    logistics_dataset = pd.read_csv("logistics_dataset.csv")
    required_columns = [
        "shipment_id", "origin_city", "origin_state", "destination_city",
        "destination_state", "fuel_cost_usd", "toll_cost_usd", "weight_lbs",
        "distance_miles"
    ]
    if not all(col in logistics_dataset.columns for col in required_columns):
        raise ValueError(f"Dataset missing required columns: {', '.join(required_columns)}")
    logger.info(f"Loaded {len(logistics_dataset)} rows from dataset")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

async def log_transcript(role: str, content: str):
    logger.info(f"[{role.upper()}]: {content}")

def get_quote_from_dataset(parsed_entities: dict) -> dict:
    """Retrieve shipment details based on ID or origin/destination."""
    if not parsed_entities:
        return {"quote_amount": None, "shipment_data": None}

    shipment = None
    shipment_id = parsed_entities.get("shipment_id")

    if shipment_id:
        shipment_id = shipment_id.upper()
        matching_shipments = logistics_dataset[logistics_dataset["shipment_id"].str.upper() == shipment_id]
        if not matching_shipments.empty:
            shipment = matching_shipments

    if (shipment is None or shipment.empty) and all(
            parsed_entities.get(k) for k in ["origin_city", "origin_state", "destination_city", "destination_state"]):
        origin_city = parsed_entities["origin_city"].lower()
        origin_state = parsed_entities["origin_state"].upper()
        destination_city = parsed_entities["destination_city"].lower()
        destination_state = parsed_entities["destination_state"].upper()

        shipment = logistics_dataset[
            (logistics_dataset["origin_city"].str.lower() == origin_city) &
            (logistics_dataset["origin_state"].str.upper() == origin_state) &
            (logistics_dataset["destination_city"].str.lower() == destination_city) &
            (logistics_dataset["destination_state"].str.upper() == destination_state)
        ]

    if shipment is not None and not shipment.empty:
        shipment = shipment.iloc[0]
        quote = (
            shipment["fuel_cost_usd"] +
            shipment["toll_cost_usd"] +
            shipment["weight_lbs"] * 0.05 +
            shipment["distance_miles"] * 2.00
        )
        return {
            "quote_amount": round(quote, 2),
            "shipment_data": shipment.to_dict()
        }

    return {"quote_amount": None, "shipment_data": None}

class LogisticsVoiceAgent(agents.Agent):
    def __init__(self, dial_info: dict):
        super().__init__(
            instructions=(
                f"You are {AGENT_NAME}, a logistics voice assistant for Qbotica, handling an outbound call. "
                "Use pause words and fillers but talk fast and naturally. "
                "Your goal is to discuss a specific shipment quote with a logistics contact. "
                "Be professional, polite, and conversational. "
                "Do NOT ask for shipment ID, origin, or destination unless the user mentions a different shipment. "
                "If the user asks about rate, weight, or cost, assume it's about the current shipment. "
                "Use the 'get_quote' tool only when a new shipment ID or location details are provided. "
                "Use the 'end_call' tool when the user explicitly requests to end the conversation. "
                "Example: 'Uh, let me check that quote for you real quick... okay, here it is!'"
            )
        )
        self.call_context = {
            'call_type': 'outbound',
            'outbound_shipment_context': dial_info.get('shipment_data', {}),
            'conversation_history': [],
            'latest_quote': dial_info.get('quote_info', {}),
            'call_active': False
        }
        self.participant = None
        self.dial_info = dial_info

    def set_participant(self, participant=None):
        self.participant = participant
        logger.info(f"Participant set: {participant.identity if participant else 'None'}")

    @agents.function_tool(name="get_quote", description="Retrieve shipment details based on shipment ID or origin/destination.")
    async def get_quote(
        self,
        ctx: agents.RunContext,
        shipment_id: Optional[str] = None,
        origin_city: Optional[str] = None,
        origin_state: Optional[str] = None,
        destination_city: Optional[str] = None,
        destination_state: Optional[str] = None
    ) -> dict:
        try:
            quote_info = get_quote_from_dataset({
                "shipment_id": shipment_id,
                "origin_city": origin_city,
                "origin_state": origin_state,
                "destination_city": destination_city,
                "destination_state": destination_state
            })
            self.call_context['latest_quote'] = quote_info
            logger.info(f"Quote retrieved: {quote_info}")

            if quote_info["quote_amount"]:
                shipment = quote_info["shipment_data"]
                response = (
                    f"Uh, here's the quote for shipment {shipment['shipment_id']} from "
                    f"{shipment['origin_city']}, {shipment['origin_state']} to "
                    f"{shipment['destination_city']}, {shipment['destination_state']}. "
                    f"It's ${quote_info['quote_amount']:.2f} for a {shipment['weight_lbs']}-pound shipment "
                    f"over {shipment['distance_miles']} miles. How's that sound?"
                )
            else:
                response = "Sorry, I couldn't find a quote with those details. Could you provide a valid shipment ID or origin and destination?"

            await log_transcript("assistant", response)
            return {"response": response}

        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            response = "I apologize, I'm having trouble retrieving that quote right now. Please try again."
            await log_transcript("assistant", response)
            return {"response": response}

    @agents.function_tool(name="end_call", description="End the conversation and disconnect the call.")
    async def end_call(self, ctx: agents.RunContext) -> dict:
        try:
            self.call_context['call_active'] = False
            self.call_context['call_type'] = 'waiting'
            logger.info("Call ended by tool")
            response = "Alright, thanks for the chat! Have a great day, goodbye!"
            await log_transcript("assistant", response)
            await asyncio.sleep(2)
            return {"response": response}

        except Exception as e:
            logger.error(f"Error ending call: {e}")
            return {"response": "Goodbye!"}

async def entrypoint(ctx: JobContext):
    health_check_task = None
    session = None
    connection_active = True

    try:
        logger.info(f"Starting {AGENT_NAME} entrypoint")
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)
        logger.info(f"Agent connected to {LIVEKIT_URL}, waiting for participant...")

        # Select a random shipment for outbound call
        if not logistics_dataset.empty:
            random_shipment = logistics_dataset.sample(n=1).iloc[0]
            quote_info = get_quote_from_dataset({
                "shipment_id": random_shipment["shipment_id"],
                "origin_city": random_shipment["origin_city"],
                "origin_state": random_shipment["origin_state"],
                "destination_city": random_shipment["destination_city"],
                "destination_state": random_shipment["destination_state"]
            })
            dial_info = {
                "phone_number": LOGISTICS_PHONE_NUMBER,
                "transfer_to": None,
                "shipment_data": random_shipment.to_dict(),
                "quote_info": quote_info
            }
        else:
            logger.error("Logistics dataset is empty, cannot initiate outbound call")
            return

        # Wait for participant with timeout
        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(),
                timeout=60.0
            )
            logger.info(f"Participant joined: {participant.identity}, kind: {participant.kind}")
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for participant")
            return
        except Exception as e:
            logger.error(f"Error waiting for participant: {e}")
            return

        # Create the agent instance
        agent = LogisticsVoiceAgent(dial_info=dial_info)
        agent.set_participant(participant)

        # Set up the agent session
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
            tts=openai.TTS(
                voice="shimmer",
                model="gpt-4o-mini-tts",
                speed=3.0,
                instructions="You are a logistics agent. Speak naturally and professionally."
            ),
            vad=silero.VAD.load()
        )

        # Update agent context for outbound call
        agent.call_context['call_type'] = 'outbound'
        agent.call_context['conversation_history'] = []
        agent.call_context['latest_quote'] = quote_info
        agent.call_context['call_active'] = True

        # Start health check
        health_check_task = asyncio.create_task(health_check())

        # Start the session
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
        )

        # Send initial greeting message
        welcome_message = (
            f"Hello, this is Qbot from Qbotica logistics. "
            f"I'm calling about shipment {dial_info['shipment_data']['shipment_id']} "
            f"from {dial_info['shipment_data']['origin_city']}, {dial_info['shipment_data']['origin_state']} "
            f"to {dial_info['shipment_data']['destination_city']}, {dial_info['shipment_data']['destination_state']}. "
            f"The current quote is ${dial_info['quote_info']['quote_amount']:.2f}. How does this rate sound to you?"
        )
        await log_transcript("assistant", welcome_message)
        await session.generate_reply(instructions=welcome_message)

        logger.info("Outbound call initiated successfully")

        # Keep the session running until it completes
        while session.turn_detection and connection_active:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in entrypoint: {e}")
        raise
    finally:
        connection_active = False
        # Cleanup tasks
        if health_check_task:
            health_check_task.cancel()
            try:
                await health_check_task
            except asyncio.CancelledError:
                pass

        # Gracefully disconnect room
        try:
            if ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                await ctx.room.disconnect()
                logger.info("LiveKit room disconnected gracefully")
        except Exception as cleanup_e:
            logger.error(f"Error during cleanup: {cleanup_e}")

async def health_check():
    """Periodic health check to ensure agent is registered with LiveKit"""
    retry_count = 0
    max_retries = 3

    while True:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{LIVEKIT_URL}/health") as resp:
                    if resp.status == 200:
                        logger.debug("Agent health check: LiveKit server is reachable")
                        retry_count = 0  # Reset retry count on success
                    else:
                        logger.warning(f"Agent health check failed: HTTP {resp.status}")
                        retry_count += 1
        except Exception as e:
            logger.error(f"Health check error: {e}")
            retry_count += 1

        if retry_count >= max_retries:
            logger.error(f"Health check failed {max_retries} times consecutively")
            # Consider breaking or implementing recovery logic

        await asyncio.sleep(30)

if __name__ == "__main__":
    logger.info(f"Starting {AGENT_NAME}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME
        )
    )