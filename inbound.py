import asyncio
import logging
import os
import pandas as pd
import json
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession, RoomInputOptions, get_job_context, BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip
from livekit.plugins import deepgram, openai, silero, noise_cancellation
from typing import Optional
import aiohttp
from faq_dataset import get_faq_data
import difflib
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LogisticsAgent")

AGENT_NAME = "inbound-agent"
LIVEKIT_ROOM_PREFIX = "call-"
ROOM_NAME = os.getenv("LIVEKIT_ROOM_NAME", f"{LIVEKIT_ROOM_PREFIX}-{random.randint(1000, 9999)}")

# Load environment variables
load_dotenv(".env")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIP_INBOUND_TRUNK_ID = os.getenv("SIP_INBOUND_TRUNK_ID")

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, OPENAI_API_KEY, SIP_INBOUND_TRUNK_ID]):
    logger.error("Missing required environment variables")
    raise ValueError("Please check .env for required API keys, LIVEKIT_URL, and SIP_INBOUND_TRUNK_ID")

# Load logistics dataset
try:
    logistics_dataset = pd.read_csv("test_dataset.csv")
    required_columns = ["shipment_id", "origin_city", "origin_state", "destination_city", "destination_state",
                        "fuel_cost_usd", "toll_cost_usd", "weight_lbs", "distance_miles", "status"]
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
    def __init__(self, room: rtc.Room):
        super().__init__(
            instructions=(
                f"You are {AGENT_NAME}, a friendly logistics assistant for Qbotica. "
                "Do not repeat the shipment ID in follow-up questions. "
                "Uhh... let me quickly check that shipment ID for you, okay? Just a sec... "
                "You help customers with shipping quotes and logistics needs. "
                "Use the 'get_quote' tool when the user provides a valid shipment ID (e.g., SHP00001) "
                "or complete origin and destination details (city and state for both, e.g., Chicago, IL to Miami, FL). "
                "For follow-up questions, reference the latest quote without calling 'get_quote' again. "
                "Use the 'get_faq' tool for general questions about logistics services. "
                "If the user asks for a joke, tell one. "
                "The 'get_quote' tool automatically sends shipment details to a Slack channel. "
                "Do not call 'send_slack_message' directly; it is handled by 'get_quote'. "
                "Be professional, helpful, and conversational. Ask for clarification if details are incomplete. "
                "Only use 'end_call' if the user explicitly requests to end the conversation."
                "If the user asks for a human agent, use 'transfer_call' to connect them. "
                "If a message is received from Slack, incorporate it into the conversation appropriately."
            )
        )
        self.call_context = {
            'call_type': 'waiting',
            'outbound_shipment_context': {},
            'conversation_history': [],
            'latest_quote': None,
            'call_active': False
        }
        self.room = room
        # Register synchronous data_received callback
        self.room.on("data_received", lambda packet: self.on_data_received(packet))

    def on_data_received(self, packet: rtc.DataPacket):
        """Handle incoming data packets from the LiveKit room, including Slack messages."""
        try:
            payload = packet.data.decode('utf-8')
            logger.debug(f"Received data packet: {payload}")
            data = json.loads(payload)
            slack_message = data.get("slack_message")
            if slack_message:
                logger.info(f"Received Slack message: {slack_message}")
                # Use asyncio.create_task to run async operations
                response = f"I received a message from our team via Slack: {slack_message}. How can I assist you further?"
                asyncio.create_task(self._process_slack_message(response))
            else:
                logger.warning(f"Invalid data packet: {payload}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse data packet: {payload}")
        except Exception as e:
            logger.error(f"Error processing data packet: {e}")

    async def _process_slack_message(self, response: str):
        """Async helper to process Slack messages and generate replies."""
        await log_transcript("assistant", response)
        await session.generate_reply(instructions=response)

    @agents.function_tool(name="get_quote",
                          description="Retrieve shipment details based on shipment ID or origin/destination.")
    async def get_quote(
            self,
            ctx: agents.RunContext,
            shipment_id: Optional[str] = None,
            origin_city: Optional[str] = None,
            origin_state: Optional[str] = None,
            destination_city: Optional[str] = None,
            destination_state: Optional[str] = None,
            status: Optional[str] = None,
            last_updated_timestamp: Optional[str] = None,
            truck_type: Optional[str] = None,
            vendor_name: Optional[str] = None,
            vendor_contact_number: Optional[str] = None
    ) -> dict:
        try:
            quote_info = get_quote_from_dataset({
                "shipment_id": shipment_id,
                "origin_city": origin_city,
                "origin_state": origin_state,
                "destination_city": destination_city,
                "destination_state": destination_state,
                "status": status,
                "last_updated_timestamp": last_updated_timestamp,
                "truck_type": truck_type,
                "vendor_name": vendor_name,
                "vendor_contact_number": vendor_contact_number
            })
            self.call_context['latest_quote'] = quote_info
            logger.info(f"Quote retrieved: {quote_info}")

            if quote_info["quote_amount"]:
                shipment = quote_info["shipment_data"]
                response = (
                    f"The quote for shipment {shipment['shipment_id']} from {shipment['origin_city']}, {shipment['origin_state']} "
                    f"to {shipment['destination_city']}, {shipment['destination_state']} is ${quote_info['quote_amount']:.2f}, "
                    f"for a {shipment['weight_lbs']}-pound shipment over {shipment['distance_miles']} miles. "
                    f"Status: {shipment['status']}. "
                    f"Last updated: {shipment['last_updated_timestamp']}. "
                    f"Truck type: {shipment['truck_type']}. "
                    f"Vendor: {shipment['vendor_name']}, Contact: {shipment['vendor_contact_number']}."
                )
                slack_message = (
                    f"Shipment query: ID {shipment['shipment_id']} from {shipment['origin_city']}, {shipment['origin_state']} "
                    f"to {shipment['destination_city']}, {shipment['destination_state']}. "
                    f"Quote: ${quote_info['quote_amount']:.2f}, Weight: {shipment['weight_lbs']} lbs, "
                    f"Distance: {shipment['distance_miles']} miles. "
                    f"Status: {shipment['status']}, Last updated: {shipment['last_updated_timestamp']}, "
                    f"Truck: {shipment['truck_type']}, Vendor: {shipment['vendor_name']} ({shipment['vendor_contact_number']})."
                )
                await self.send_slack_message(ctx, message=slack_message)
            else:
                response = "No quote found for the provided details. Could you verify the shipment ID or provide origin and destination details?"

            await log_transcript("assistant", response)
            return {"response": response}

        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            response = f"I apologize, but I'm having trouble retrieving that quote right now. Please try again."
            await log_transcript("assistant", response)
            return {"response": response}

    @agents.function_tool(name="send_slack_message", description="Send a message to a Slack channel via the room.")
    async def send_slack_message(self, ctx: agents.RunContext, message: str) -> dict:
        try:
            data = {"message": message}
            payload = json.dumps(data).encode('utf-8')
            await self.room.local_participant.publish_data(
                payload=payload
            )
            logger.info(f"Published Slack message to room message={message}")
            return {"status": "success", "response": "Message sent to Slack agent."}
        except Exception as e:
            logger.error(f"Error publishing Slack message: {e}")
            return {"status": "failed", "response": f"Failed to send message: {str(e)}"}

    @agents.function_tool(name="get_faq",
                          description="Retrieve answers to general FAQ questions about logistics services.")
    async def get_faq(self, ctx: agents.RunContext, question: str) -> dict:
        try:
            question = question.lower().strip()
            answer = get_faq_answer(question)
            if answer:
                response = answer
                logger.info(f"FAQ answer retrieved: {response}")
            else:
                response = "I'm sorry, I don't have an answer for that question. Could you provide more details or ask something else?"
                logger.info("No FAQ answer found for question")

            await log_transcript("assistant", response)
            return {"response": response}

        except Exception as e:
            logger.error(f"Error getting FAQ answer: {e}")
            response = "I apologize, but I'm having trouble answering that question right now. Please try again."
            await log_transcript("assistant", response)
            return {"response": response}

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @agents.function_tool(name="transfer_call", description="Transfer the call to a human agent.")
    async def transfer_call(self, ctx: agents.RunContext) -> dict:
        """Transfer the call to a human agent after confirming with the user."""
        transfer_to = os.getenv("TRANSFER_PHONE_NUMBER", "+16023016597")
        if not transfer_to:
            logger.error("No transfer phone number configured")
            response = "I'm sorry, I cannot transfer the call at this time. Please try again later."
            await log_transcript("assistant", response)
            return {"response": response}

        logger.info(f"Initiating call transfer to {transfer_to}")

        try:
            response = "Please hold while I transfer you to a human agent."
            await log_transcript("assistant", response)
            await session.generate_reply(instructions=response)

            await asyncio.sleep(2)

            participants = await ctx.room.list_participants()
            if not participants:
                logger.error("No participants found in the room")
                response = "I'm sorry, I encountered an issue while trying to transfer the call."
                await log_transcript("assistant", response)
                return {"response": response}

            participant = next((p for p in participants if p.identity != AGENT_NAME), None)
            if not participant:
                logger.error("No valid participant found for transfer")
                response = "I'm sorry, I encountered an issue while trying to transfer the call."
                await log_transcript("assistant", response)
                return {"response": response}

            await ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=ctx.room.name,
                    participant_identity=participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )

            logger.info(f"Successfully transferred call to {transfer_to}")
            response = "Transfer successful. You are now being connected to a human agent."
            await log_transcript("assistant", response)
            return {"response": response}

        except Exception as e:
            logger.error(f"Error transferring call: {e}")
            response = "I'm sorry, there was an issue transferring your call. How else can I assist you?"
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
            await session.generate_reply(instructions=response)

            await asyncio.sleep(2)
            await self.hangup()
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

def get_faq_answer(question: str) -> Optional[str]:
    question = question.lower().strip()
    faq_data = get_faq_data()
    questions = list(faq_data.keys())
    matches = difflib.get_close_matches(question, questions, n=1, cutoff=0.6)
    if matches:
        return faq_data[matches[0]]
    return None

async def entrypoint(ctx: JobContext):
    health_check_task = None
    global session
    session = None
    connection_active = True
    inactivity_timeout = 300
    last_activity_time = asyncio.get_event_loop().time()

    try:
        logger.info(f"Starting {AGENT_NAME} entrypoint for room: {ctx.room.name}")
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)
        logger.info(f"Agent connected to {LIVEKIT_URL}, waiting for participant...")

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

        agent = LogisticsAgent(room=ctx.room)
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

        agent.call_context['call_type'] = 'inbound'
        agent.call_context['conversation_history'] = []
        agent.call_context['latest_quote'] = None
        agent.call_context['call_active'] = True

        health_check_task = asyncio.create_task(health_check())
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
        )

        welcome_message = (
            f"Hello! This is Qbot from Qbotica logistics. "
            "I'm here to help you with your shipping needs, provide quotes, or answer general questions. "
            "How can I assist you today?"
        )
        await log_transcript("assistant", welcome_message)
        await session.generate_reply(instructions=welcome_message)
        last_activity_time = asyncio.get_event_loop().time()

        logger.info("Inbound call initiated successfully")

        # background_audio = BackgroundAudioPlayer(
        #     thinking_sound=[
        #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
        #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        #     ],
        # )
        # await background_audio.start(room=ctx.room, agent_session=session)

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

        if health_check_task:
            health_check_task.cancel()
            try:
                await health_check_task
            except asyncio.CancelledError:
                pass

        try:
            if ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                await ctx.room.disconnect()
                logger.info("LiveKit room disconnected gracefully")
        except Exception as cleanup_e:
            logger.error(f"Error during cleanup: {cleanup_e}")

async def health_check():
    retry_count = 0
    max_retries = 3

    while True:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{LIVEKIT_URL}/health") as resp:
                    if resp.status == 200:
                        logger.debug("Agent health check: LiveKit server is reachable")
                        retry_count = 0
                    else:
                        logger.warning(f"Agent health check failed: HTTP {resp.status}")
                        retry_count += 1
        except Exception as e:
            logger.error(f"Health check error: {e}")
            retry_count += 1

        if retry_count >= max_retries:
            logger.error(f"Health check failed {max_retries} times consecutively")
        await asyncio.sleep(30)

if __name__ == "__main__":
    logger.info(f"Starting {AGENT_NAME} for SIP trunk {SIP_INBOUND_TRUNK_ID}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME
        )
    )