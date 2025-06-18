import asyncio
import logging
import os
import pandas as pd
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, llm, Agent, AgentSession, RoomInputOptions
from livekit.plugins import deepgram, openai, silero, noise_cancellation, elevenlabs
from typing import Optional
import aiohttp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LogisticsAgent")

AGENT_NAME = "inbound-agent"
LIVEKIT_ROOM_PREFIX = "call"

# Load environment variables
load_dotenv(".env")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIP_INBOUND_TRUNK_ID = os.getenv("SIP_INBOUND_TRUNK_ID", "ST_8nUkUTsN5xV4")

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, OPENAI_API_KEY, SIP_INBOUND_TRUNK_ID]):
    logger.error("Missing required environment variables")
    raise ValueError("Please check .env for required API keys, LIVEKIT_URL, and SIP_INBOUND_TRUNK_ID")

# Load logistics dataset
try:
    logistics_dataset = pd.read_csv("logistics_dataset.csv")
    required_columns = ["shipment_id", "origin_city", "origin_state", "destination_city", "destination_state",
                        "fuel_cost_usd", "toll_cost_usd", "weight_lbs", "distance_miles"]
    if not all(col in logistics_dataset.columns for col in required_columns):
        logger.error("Dataset missing required columns")
        raise ValueError("Required columns: " + ", ".join(required_columns))
    logger.info(f"Loaded {len(logistics_dataset)} rows from dataset")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

async def log_transcript(role: str, content: str):
    logger.info(f"[{role.upper()}]: {content}")

class LogisticsAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                f"You are {AGENT_NAME}, a friendly logistics assistant for Qbotica. "
                "Use pause words and fillers but talk fast more naturally in your speech. "
                "Uhh... let me quickly check that shipment ID for you, okay? Just a sec..."
                "You help customers with shipping quotes and logistics needs. "
                "Use the 'get_quote' tool when the user provides a valid shipment ID (e.g., SHP00001) "
                "or complete origin and destination details (city and state for both, e.g., Chicago, IL to Miami, FL). "
                "For follow-up questions, reference the latest quote without calling 'get_quote' again. "
                "If the user asks to tell a joke, please do so. "
                "Be professional, helpful, and conversational. Ask for clarification if details are incomplete. "
                "Only use 'end_call' if the user explicitly requests to end the conversation."
            )
        )
        self.call_context = {
            'call_type': 'waiting',
            'outbound_shipment_context': {},
            'conversation_history': [],
            'latest_quote': None,
            'call_active': False
        }

    @agents.function_tool(name="get_quote",
                          description="Retrieve shipment details based on shipment ID or origin/destination.")
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
                    f"The quote for shipment {shipment['shipment_id']} from {shipment['origin_city']}, {shipment['origin_state']} "
                    f"to {shipment['destination_city']}, {shipment['destination_state']} is ${quote_info['quote_amount']}, "
                    f"for a {shipment['weight_lbs']}-pound shipment over {shipment['distance_miles']} miles."
                )
            else:
                response = "No quote found for the provided details."

            await log_transcript("assistant", response)
            return {"response": response}

        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            response = f"I apologize, but I'm having trouble retrieving that quote right now. Please try again."
            await log_transcript("assistant", response)
            return {"response": response}

    @agents.function_tool(name="end_call", description="End the conversation and disconnect the call.")
    async def end_call(self, ctx: agents.RunContext) -> dict:
        try:
            self.call_context['call_active'] = False
            self.call_context['call_type'] = 'waiting'
            logger.info("Call ended by tool")
            response = "Thank you for calling Qbotica logistics. Have a great day! Goodbye!"
            await log_transcript("assistant", response)

            # Add a small delay to ensure the goodbye message is delivered
            await asyncio.sleep(2)

            # Gracefully end the call without deleting the room
            # The room will be cleaned up automatically by the entrypoint
            return {"response": response}

        except Exception as e:
            logger.error(f"Error ending call: {e}")
            return {"response": "Goodbye!"}

def get_quote_from_dataset(parsed_entities: dict) -> dict:
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

async def entrypoint(ctx: JobContext):
    health_check_task = None
    session = None
    connection_active = True
    inactivity_timeout = 300  # 5-minute inactivity timeout in seconds
    last_activity_time = asyncio.get_event_loop().time()

    try:
        logger.info(f"Starting {AGENT_NAME} entrypoint for room: {ctx.room.name}")
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)
        logger.info(f"Agent connected to {LIVEKIT_URL}, waiting for participant...")

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
        agent = LogisticsAgent()

        # Set up the agent session with correct parameters
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
            tts=deepgram.TTS(
                model="aura-2-andromeda-en"
            ),
            vad=silero.VAD.load()
        )

        # Update agent context for inbound call
        agent.call_context['call_type'] = 'inbound'
        agent.call_context['conversation_history'] = []
        agent.call_context['latest_quote'] = None
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
            f"Hello! This is Qbot from Qbotica logistics. "
            "I'm here to help you with your shipping needs and provide quotes. "
            "How can I assist you today?"
        )
        await log_transcript("assistant", welcome_message)
        await session.generate_reply(instructions=welcome_message)
        last_activity_time = asyncio.get_event_loop().time()

        logger.info("Inbound call initiated successfully")

        # Keep the session running until it completes or times out
        while connection_active:
            if session.turn_detection:
                last_activity_time = asyncio.get_event_loop().time()
            elif asyncio.get_event_loop().time() - last_activity_time > inactivity_timeout:
                logger.info("Inactivity timeout reached, ending call")
                await agent.end_call(ctx)
                break
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
    logger.info(f"Starting {AGENT_NAME} for SIP trunk {SIP_INBOUND_TRUNK_ID}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME
        )
    )