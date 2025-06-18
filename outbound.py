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
load_dotenv(".env")
required_env = [
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY", "OPENAI_API_KEY", "LOGISTICS_PHONE_NUMBER"
]
for var in required_env:
    if not os.getenv(var):
        raise ValueError(f"{var} missing in .env")

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
        "origin_city", "origin_state", "destination_city",
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
    """Retrieve load details based on origin/destination."""
    if not parsed_entities:
        return {"quote_amount": None, "load_data": None}

    load = None
    if all(parsed_entities.get(k) for k in ["origin_city", "origin_state", "destination_city", "destination_state"]):
        origin_city = parsed_entities["origin_city"].lower()
        origin_state = parsed_entities["origin_state"].upper()
        destination_city = parsed_entities["destination_city"].lower()
        destination_state = parsed_entities["destination_state"].upper()

        load = logistics_dataset[
            (logistics_dataset["origin_city"].str.lower() == origin_city) &
            (logistics_dataset["origin_state"].str.upper() == origin_state) &
            (logistics_dataset["destination_city"].str.lower() == destination_city) &
            (logistics_dataset["destination_state"].str.upper() == destination_state)
        ]

    if load is not None and not load.empty:
        load = load.iloc[0]
        base_cost = (
            load["fuel_cost_usd"] +
            load["toll_cost_usd"] +
            load["weight_lbs"] * 0.05 +
            load["distance_miles"] * 2.00
        )
        # Initial quote with 25% markup for profit
        quote = base_cost * 1.25
        return {
            "quote_amount": round(quote, 2),
            "base_cost": round(base_cost, 2),
            "load_data": load.to_dict()
        }

    return {"quote_amount": None, "base_cost": None, "load_data": None}

class LogisticsVoiceAgent(agents.Agent):
    def __init__(self, dial_info: dict):
        super().__init__(
            instructions=(
                f"You are {AGENT_NAME}, a logistics negotiator for Qbotica, handling outbound calls to logistics companies. "
                "Your goal is to negotiate and confirm the best price and sending date for sending a specific load. "
                "Use pause words and fillers, talk fast and naturally, and be persuasive yet polite. "
                "Start with the load context in the welcome message and focus on negotiating the price (starting with a 25% profit margin) "
                "down to a minimum 15% profit margin, while confirming the sending date. "
                "Do not mention other loads or the word 'shipment.' "
                "Use the 'get_quote' tool only for new load details if provided. "
                "Use the 'end_call' tool when the price and date are agreed or the user ends the call. "
                "Example: 'Uh, let’s start at $500—can we get it to $450 and confirm a date?'"
            )
        )
        self.call_context = {
            'call_type': 'outbound',
            'outbound_load_context': dial_info.get('load_data', {}),
            'conversation_history': [],
            'latest_quote': dial_info.get('quote_info', {}),
            'call_active': False
        }
        self.participant = None
        self.dial_info = dial_info

    def set_participant(self, participant=None):
        self.participant = participant
        logger.info(f"Participant set: {participant.identity if participant else 'None'}")

    @agents.function_tool(name="get_quote", description="Retrieve load details based on origin/destination.")
    async def get_quote(
        self,
        ctx: agents.RunContext,
        origin_city: Optional[str] = None,
        origin_state: Optional[str] = None,
        destination_city: Optional[str] = None,
        destination_state: Optional[str] = None
    ) -> dict:
        try:
            quote_info = get_quote_from_dataset({
                "origin_city": origin_city,
                "origin_state": origin_state,
                "destination_city": destination_city,
                "destination_state": destination_state
            })
            self.call_context['latest_quote'] = quote_info
            logger.info(f"Quote retrieved: {quote_info}")

            if quote_info["quote_amount"]:
                load = quote_info["load_data"]
                base_cost = quote_info["base_cost"]
                initial_quote = quote_info["quote_amount"]
                # Minimum quote with 15% profit margin
                min_quote = base_cost * 1.15
                negotiated_quote = max(min_quote, initial_quote * 0.9)  # Ensure at least 15% profit
                # Natural follow-up and negotiation variations with date confirmation
                follow_ups = [
                    f"Uh, let’s start at ${initial_quote:.2f}—how about we bring it down to ${negotiated_quote:.2f}? Can you confirm a sending date?",
                    f"Cool, I’m proposing ${initial_quote:.2f}—can we settle at ${negotiated_quote:.2f} and pick a date?",
                    f"Okay, let’s begin with ${initial_quote:.2f}—what about ${negotiated_quote:.2f} with a confirmed date?",
                    f"Great, how about ${initial_quote:.2f} to start? Let’s negotiate to ${negotiated_quote:.2f} and set a date!"
                ]
                follow_up = random.choice(follow_ups)
                response = (
                    f"The quote for this load from "
                    f"{load['origin_city']}, {load['origin_state']} to "
                    f"{load['destination_city']}, {load['destination_state']} is "
                    f"${initial_quote:.2f} for a {load['weight_lbs']}-pound load "
                    f"over {load['distance_miles']} miles. {follow_up}"
                )
            else:
                response = "Sorry, I couldn’t find details for this. Could you provide the origin and destination?"

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
            response = "Alright, we’ve agreed on the price and date—thanks for sorting this out! Goodbye!"
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

        # Select a random load for outbound call
        if not logistics_dataset.empty:
            random_load = logistics_dataset.sample(n=1).iloc[0]
            quote_info = get_quote_from_dataset({
                "origin_city": random_load["origin_city"],
                "origin_state": random_load["origin_state"],
                "destination_city": random_load["destination_city"],
                "destination_state": random_load["destination_state"]
            })
            dial_info = {
                "phone_number": LOGISTICS_PHONE_NUMBER,
                "transfer_to": None,
                "load_data": random_load.to_dict(),
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
            stt=openai.STT(
                model="whisper-1",
                noise_reduction_type="near_feild"
            ),
            llm=openai.LLM(model="gpt-4o", temperature=0.7),
            tts=deepgram.TTS(
                model="aura-2-andromeda-en"
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
            f"I'm calling to discuss sending this load from "
            f"{dial_info['load_data']['origin_city']}, {dial_info['load_data']['origin_state']} to "
            f"{dial_info['load_data']['destination_city']}, {dial_info['load_data']['destination_state']}. "
            "Is it a good time to talk about the price and sending date?"
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