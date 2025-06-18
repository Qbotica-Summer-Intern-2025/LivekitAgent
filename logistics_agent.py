import asyncio
import json
import os
import pandas
import logging
import aiohttp.web
import aiohttp
import time
import jwt
from livekit import rtc
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents, AgentWebSocketClient
from openai import OpenAI
from dotenv import load_dotenv
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")

_http_session = None
AGENT_NAME = "Qbot"

load_dotenv(".env.local")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
LIVEKIT_ROOM = "playground-B6Mt-CCXH"

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY]):
    logger.error("Missing required environment variables")
    raise ValueError(
        "Please check .env.local for LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, OPENAI_API_KEY, and CARTESIA_API_KEY")

dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    logistics_dataset = pandas.read_csv("logistics_dataset.csv")
    required_columns = ["shipment_id", "origin_city", "origin_state", "destination_city", "destination_state",
                        "fuel_cost_usd", "toll_cost_usd", "weight_lbs", "distance_miles"]
    if not all(col in logistics_dataset.columns for col in required_columns):
        logger.error("Dataset missing required columns")
        raise ValueError("Dataset must contain columns: " + ", ".join(required_columns))
    logger.info(f"Loaded {len(logistics_dataset)} rows from dataset")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

_agent_room: rtc.Room = None
_agent_audio_source: rtc.AudioSource = None
_agent_audio_track: rtc.LocalAudioTrack = None
_current_call_context = {
    'call_type': 'inbound',
    'outbound_shipment_context': {},
    'conversation_history': [],
    'latest_quote': None
}
_room_ready = asyncio.Event()

ui_websockets = set()

async def broadcast_to_ui(message_type: str, text: str):
    message = json.dumps({"type": message_type, "text": text})
    for ws in list(ui_websockets):
        try:
            await ws.send_str(message)
        except Exception as e:
            logger.error(f"Failed to send to UI WebSocket: {e}")
            ui_websockets.discard(ws)

async def initiate_inbound_call():
    global _current_call_context
    _current_call_context['call_type'] = 'inbound'
    _current_call_context['outbound_shipment_context'] = {}
    _current_call_context['latest_quote'] = None
    _current_call_context['conversation_history'] = []
    welcome_message = await final_resp("Start an inbound call with a welcome message.", {})
    await broadcast_to_ui("system", welcome_message)
    audio_data = await generate_audio(welcome_message)
    if audio_data:
        await send_audio(audio_data)
    else:
        logger.warning("Failed to generate inbound audio.")
    _current_call_context['conversation_history'].append({"role": "assistant", "content": welcome_message})

async def initiate_outbound_call():
    global _current_call_context
    if logistics_dataset.empty:
        error_message = await final_resp("No shipments available for an outbound call.", {})
        await broadcast_to_ui("system", error_message)
        return
    random_shipment = logistics_dataset.sample(n=1).iloc[0]
    quote_details = get_quote_from_dataset({
        "shipment_id": random_shipment["shipment_id"],
        "origin_city": random_shipment["origin_city"],
        "origin_state": random_shipment["origin_state"],
        "destination_city": random_shipment["destination_city"],
        "destination_state": random_shipment["destination_state"]
    })
    initial_quote_amount = quote_details.get("quote_amount")
    _current_call_context['outbound_shipment_context'] = {
        "shipment_id": str(random_shipment["shipment_id"]),
        "origin_city": str(random_shipment["origin_city"]),
        "origin_state": str(random_shipment["origin_state"]),
        "destination_city": str(random_shipment["destination_city"]),
        "destination_state": str(random_shipment["destination_state"]),
        "weight_lbs": float(random_shipment["weight_lbs"]),
        "distance_miles": float(random_shipment["distance_miles"]),
        "quote_amount": initial_quote_amount,
    }
    _current_call_context['latest_quote'] = quote_details
    _current_call_context['conversation_history'] = []
    initial_message = await final_resp(
        f"Generate a short, professional outbound call introduction for a demo. You are {AGENT_NAME} from Qbotica, calling a logistics company to get a better quote for shipment {random_shipment['shipment_id']} from {random_shipment['origin_city']}, {random_shipment['origin_state']} to {random_shipment['destination_city']}, {random_shipment['destination_state']}. Keep it concise, avoid placeholders like client name, and focus on requesting a quote.",
        quote_details
    )
    logger.info(f"Outbound message: {initial_message}")
    await broadcast_to_ui("system", initial_message)
    audio_data = await generate_audio(initial_message)
    if audio_data:
        await send_audio(audio_data)
    else:
        logger.warning("Failed to generate outbound audio.")
    _current_call_context['conversation_history'].append({"role": "assistant", "content": initial_message})
    _current_call_context['call_type'] = 'inbound'

async def ui_websocket(request: aiohttp.web.Request):
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("UI WebSocket connected")
    ui_websockets.add(ws)
    try:
        async for msg in ws:
            if msg.type == aiohttp.web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message_type = data.get("type")
                    if message_type == "user_text" and data.get("text"):
                        logger.info(f"Received user text from UI: {data['text']}")
                        await handle_transcript(None, data["text"])
                    elif message_type == "init_inbound":
                        logger.info("Received init_inbound request")
                        await initiate_inbound_call()
                    elif message_type == "init_outbound":
                        logger.info("Received init_outbound request")
                        await initiate_outbound_call()
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in WebSocket message: {msg.data}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
            elif msg.type == aiohttp.web.WSMsgType.ERROR:
                logger.error(f"UI WebSocket error: {ws.exception()}")
    except Exception as e:
        logger.error(f"UI WebSocket handler error: {e}")
    finally:
        logger.info("UI WebSocket disconnected")
        ui_websockets.discard(ws)
    return ws

def create_jwt(cid: str, for_playground: bool = False) -> str:
    try:
        payload = {
            "iss": LIVEKIT_API_KEY,
            "sub": f"{cid}{' (playground)' if for_playground else ''}",
            "exp": int(time.time()) + 3600,
            "video": {
                "roomJoin": True,
                "room": LIVEKIT_ROOM,
                "roomCreate": True,
                "canPublish": True,
                "canSubscribe": True
            },
            "name": f"{AGENT_NAME} Agent" if not for_playground else "Playground Test Agent"
        }
        token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
        token_type = "Playground" if for_playground else "Agent"
        logger.info(f"Generated {token_type} JWT for identity={payload['sub']}, room={LIVEKIT_ROOM}")
        return token
    except Exception as e:
        logger.error(f"Failed to generate JWT: {e}")
        raise

def split_text(text: str, max_words: int = 50) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_word_count = 0
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        if current_word_count + word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

async def generate_audio(text: str) -> bytes | None:
    global _http_session
    start_time = time.time()
    if _http_session is None:
        logger.error("HTTP session not initialized")
        return None
    word_count = len(text.split())
    logger.info(f"Generating audio for text ({word_count} words): {text[:50]}...")
    retries = 2
    for attempt in range(retries):
        try:
            headers = {
                "Cartesia-Version": "2024-06-10",
                "X-API-Key": CARTESIA_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "transcript": str(text),
                "model_id": "sonic-english",
                "voice": {"mode": "id", "id": "0c8ed86e-6c64-40f0-b252-b773911de6bb"},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000
                }
            }
            async with _http_session.post(
                    "https://api.cartesia.ai/tts/bytes",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15.0)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Cartesia API error (attempt {attempt+1}/{retries}): {response.status} - {error_text}, took {time.time() - start_time:.2f}s")
                    continue
                audio = await response.read()
                logger.info(f"Generated audio data length: {len(audio)}, took {time.time() - start_time:.2f}s")
                return audio
        except aiohttp.ClientError as e:
            logger.error(f"Cartesia TTS error (attempt {attempt+1}/{retries}): {type(e).__name__}: {str(e)}, took {time.time() - start_time:.2f}s")
        except asyncio.TimeoutError as e:
            logger.error(f"Cartesia TTS timeout after 15s: {str(e)}, total time {time.time() - took:.2f}s")
        except Exception as e:
            logger.error(f"Error (attempt {type(e)}/{str(e)}): {str(e)}, took {time.time() - start_time:.2f}s")
        if attempt < retries - 1:
            await asyncio.sleep(1)
    logger.error(f"Failed to generate audio after {retries} attempts")
    return None

async def send_audio(audio_data: bytes):
    global _agent_audio_source
    if not audio_data or not _agent_audio_source:
        logger.warning("No audio data or audio source unavailable")
        return
    try:
        samples_per_channel = len(audio_data) // 2
        audio_frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples_per_channel
        )
        try:
            await _agent_audio_source.capture_frame(audio_frame)
            logger.info("Audio sent successfully using capture_frame")
        except AttributeError:
            try:
                await _agent_audio_source.write_frame(audio_frame)
                logger.info("Audio sent successfully using write_frame")
            except AttributeError:
                if hasattr(_agent_audio_source, 'publish_data'):
                    await _agent_audio_source.publish_data(audio_data)
                    logger.info("Audio sent successfully using publish_data")
                else:
                    logger.error("No compatible audio sending method found")
    except Exception as e:
        logger.error(f"Failed to send audio: {e}")

def get_quote_from_dataset(parsed_entities: dict) -> dict:
    if not parsed_entities:
        return {"quote_amount": None, "shipment_data": None}
    shipment = None
    shipment_id = parsed_entities.get("shipment_id")
    if shipment_id:
        shipment_id = shipment_id.upper()
        shipment = logistics_dataset[logistics_dataset["shipment_id"].str.upper() == shipment_id]
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
            "quote_amount": quote,
            "shipment_data": shipment.to_dict()
        }
    return {"quote_amount": None, "shipment_data": None}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Retrieve shipment details based on shipment ID or origin/destination to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment_id": {"type": "string", "description": "Shipment ID (e.g., SHP03936)"},
                    "origin_city": {"type": "string", "description": "Origin city"},
                    "origin_state": {"type": "string", "description": "Origin state code (e.g., TX)"},
                    "destination_city": {"type": "string", "description": "Destination city"},
                    "destination_state": {"type": "string", "description": "Destination state code (e.g., AZ)"}
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the conversation and disconnect the call.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

async def process_user_input_with_llm(text: str, current_conversation_context: dict) -> dict:
    try:
        system_prompt = f"""
You are {AGENT_NAME}, a logistics assistant for Qbotica, specializing in shipment quote queries. Be polite, professional, and conversational, responding like a human agent. Use the 'get_quote' tool only when the user provides a shipment ID or origin/destination details to retrieve a new quote.And keep the conversation concise. For follow-up questions (e.g., 'What's the weight?', 'Is there a better price?'), use the latest shipment details if available, without calling 'get_quote' again. Handle negotiation questions (e.g., 'best price', 'discount') by referencing the current quote and explaining pricing naturally or offering to explore alternatives. Respond naturally to greetings, affirmations, or unrelated inputs, and ask for clarification if needed. Use the 'end_call' tool only when the user explicitly requests to end the conversation. If no shipment details are available, guide the user to provide them in a friendly way.
"""
        current_conversation_context['conversation_history'].append({"role": "user", "content": text})
        filtered_history = []
        i = 0
        while i < len(current_conversation_context['conversation_history']):
            msg = current_conversation_context['conversation_history'][i]
            if msg['role'] == 'tool':
                if i > 0 and 'tool_calls' in current_conversation_context['conversation_history'][i-1]:
                    filtered_history.append(msg)
                i += 1
            else:
                filtered_history.append(msg)
                i += 1
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                *filtered_history
            ],
            tools=TOOLS,
            tool_choice="auto"
        )
        response = completion.choices[0].message
        logger.info(f"OpenAI response: {json.dumps(response.to_dict())}")
        response_text = response.content or ""
        tool_calls = response.tool_calls or []
        if response_text:
            current_conversation_context['conversation_history'].append({"role": "assistant", "content": response_text})
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid tool call arguments: {tool_call.function.arguments}, error: {e}")
                response_text = await final_resp(text, current_conversation_context.get('latest_quote', {}))
                current_conversation_context['conversation_history'].append({
                    "role": "assistant",
                    "content": response_text
                })
                return {"response": response_text, "end_call": False}
            if function_name == "get_quote":
                quote_info = get_quote_from_dataset(args)
                current_conversation_context['latest_quote'] = quote_info
                response_text = await final_resp(text, quote_info)
                current_conversation_context['conversation_history'].append({
                    "role": "tool",
                    "content": json.dumps(quote_info),
                    "tool_call_id": tool_call.id
                })
                current_conversation_context['conversation_history'].append({
                    "role": "assistant",
                    "content": response_text
                })
            elif function_name == "end_call":
                response_text = await final_resp("End the call politely.", {})
                current_conversation_context['conversation_history'].append({
                    "role": "tool",
                    "content": response_text,
                    "tool_call_id": tool_call.id
                })
                return {"response": response_text, "end_call": True}
        if not tool_calls:
            response_text = await final_resp(text, current_conversation_context.get('latest_quote', {}))
            current_conversation_context['conversation_history'].append({
                "role": "assistant",
                "content": response_text
            })
        return {"response": response_text, "end_call": False}
    except Exception as e:
        logger.error(f"Error processing OpenAI API request: {type(e).__name__}: {str(e)}")
        response_text = await final_resp(f"Handle an error: {str(e)}", current_conversation_context.get('latest_quote', {}))
        current_conversation_context['conversation_history'].append({
            "role": "assistant",
            "content": response_text
        })
        return {"response": response_text, "end_call": False}

async def final_resp(user_question: str, quote_info: dict) -> str:
    try:
        shipment_data = quote_info.get("shipment_data", {})
        quote_amount = quote_info.get("quote_amount")
        prompt = f"""
You are {AGENT_NAME}, a logistics assistant for Qbotica. Respond to the user's question in a polite, professional, and conversational manner, as a human would. Use the provided shipment data to answer specific queries (e.g., weight, distance, quote amount). And keep the conversation concise. For negotiation questions (e.g., 'best price', 'discount'), reference the quote amount and explain pricing naturally, suggesting alternatives if appropriate. For greetings, affirmations, or unrelated inputs, respond warmly and appropriately. If the question is vague or shipment data is missing, ask for clarification in a friendly way. If generating a welcome message or handling errors, craft a natural response that fits the context. Avoid robotic or formulaic replies. If asked for your name, respond with '{AGENT_NAME}' and mention you're with Qbotica.

Shipment Data: {json.dumps(shipment_data, default=str)}
Quote Amount: {quote_amount if quote_amount is not None else 'Not available'}
Conversation History: {json.dumps(_current_call_context['conversation_history'], default=str)}

User Question: {user_question}

Answer:
"""
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        response_text = completion.choices[0].message.content.strip()
        logger.info(f"Generated response for question '{user_question}': {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"Failed to generate response: {type(e).__name__}: {str(e)}")
        return f"I'm sorry, {AGENT_NAME} here from Qbotica. Something went wrong. Could you please try again or provide more details?"

async def handle_transcript(participant: rtc.RemoteParticipant | None, text: str):
    start_time = time.time()
    global _agent_room, _agent_audio_source, _current_call_context
    try:
        await asyncio.wait_for(_room_ready.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("LiveKit room not ready after 10s")
        error_response = await final_resp("System timeout error.", {})
        await broadcast_to_ui("system", error_response)
        return
    if text:
        logger.debug(f"Processing text: {text} (from {'UI' if participant is None else participant.identity})")
        await broadcast_to_ui("user", text)
    try:
        llm_response = await process_user_input_with_llm(text, _current_call_context)
        logger.info(f"OpenAI processing took {time.time() - start_time:.2f}s")
        response_text = llm_response["response"]
        logger.info(f"Response Text: {response_text}")
        if response_text:
            await broadcast_to_ui("response", response_text)
            audio_start = time.time()
            text_chunks = split_text(response_text, max_words=50)
            logger.info(f"Split response into {len(text_chunks)} chunks for audio generation")
            for chunk in text_chunks:
                audio_data = await generate_audio(chunk)
                if audio_data:
                    await send_audio(audio_data)
                    logger.debug(f"Sent audio for chunk: {chunk[:50]}...")
                else:
                    logger.warning(f"Failed to generate audio for chunk: {chunk[:50]}...")
                    error_response = await final_resp(f"Audio generation failed for part of the response.", {})
                    await broadcast_to_ui("system", error_response)
                    break
            logger.info(f"Audio generation and sending took {time.time() - audio_start:.2f}s")
        if llm_response.get("end_call"):
            if _agent_room:
                await _agent_room.disconnect()
            logger.info("Call ended by user request.")
    except Exception as e:
        logger.error(f"Error in handle_transcript: {type(e).__name__}: {str(e)}, total time {time.time() - start_time:.2f}s")
        error_response = await final_resp(f"Handle an error: {str(e)}", {})
        await broadcast_to_ui("system", error_response)
    logger.info(f"Handle transcript total time: {time.time() - start_time:.2f}s")

async def on_track_subscribed(track: rtc.RemoteTrack, publication: rtc.RemoteTrackPublication,
                              participant: rtc.RemoteParticipant):
    global _agent_room
    logger.info(f"Subscribed to track from {participant.identity} - type: {track.kind}")
    if track.kind != rtc.TrackKind.KIND_AUDIO:
        logger.warning(f"Track from {participant.identity} is not audio. Skipping.")
        return
    logger.info(f"Audio track received from {participant.identity}. Start Deepgram...")
    try:
        dg_connection = dg_client.listen.websocket.v("1")

        def on_message(self, data, **kwargs):
            try:
                transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "").strip()
                if transcript:
                    logger.info(f"Transcript from {participant.identity}: {transcript}")
                    asyncio.create_task(handle_transcript(participant, transcript))
            except Exception as e:
                logger.error(f"Failed to process transcription: {e}")

        def on_error(self, error, **kwargs):
            logger.error(f"Deepgram error for {participant.identity}: {error}")

        dg_connection.on("message", on_message)
        dg_connection.on("error", on_error)

        options = {
            "model": "nova-2",
            "encoding": "linear16",
            "sample_rate": 16000,
            "language": "en-US",
            "interim_results": True,
            "smart_format": True,
            "endpointing": 300
        }

        if not dg_connection.start(**options):
            logger.error("Failed to start Deepgram WebSocket connection")
            return

        if not hasattr(_agent_room, 'active_transcriptions'):
            _agent_room.active_transcriptions = {}
        _agent_room.active_transcriptions[participant.identity] = dg_connection
        logger.info(f"Deepgram transcription started for {participant.identity}")

        async def forward_audio():
            try:
                audio_stream = rtc.AudioStream(track)
                async for frame_event in audio_stream:
                    pcm_data = frame_event.frame.data
                    dg_connection.send(bytes(pcm_data))  # 🔥 FIXED
            except Exception as e:
                logger.error(f"Audio forwarding failed for {participant.identity}: {e}")
            finally:
                dg_connection.finish()
                logger.info(f"Deepgram transcription closed for {participant.identity}")

        asyncio.create_task(forward_audio())
    except Exception as e:
        logger.error(f"Failed to start Deepgram for {participant.identity}: {e}")


async def handle_rpc(room: rtc.Room, data: dict):
    logger.info(f"Received RPC: {data}")
    return {"status": "ok", "message": "RPC not implemented."}

async def handle_room_connection():
    global _agent_room, _agent_audio_source, _agent_audio_track, _current_call_context, _room_ready
    _agent_room = rtc.Room()
    _agent_room.active_transcriptions = {}
    def track_subscribed_wrapper(track, publication, participant):
        asyncio.create_task(on_track_subscribed(track, publication, participant))
    def rpc_handler_wrapper(data):
        asyncio.create_task(handle_rpc(_agent_room, data))
    _agent_room.on("track_subscribed", track_subscribed_wrapper)
    _agent_room.on("rpc", rpc_handler_wrapper)
    agent_token = create_jwt("logistics-agent")
    playground_token = create_jwt("playground-user", for_playground=True)
    try:
        await _agent_room.connect(LIVEKIT_URL, agent_token)
        logger.info("Connected to LiveKit room with agent token")
        _agent_audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
        _agent_audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent_response_audio",
            source=_agent_audio_source
        )
        await _agent_room.local_participant.publish_track(_agent_audio_track)
        logger.info("Agent's audio track published")
        _room_ready.set()
        await initiate_inbound_call()
        logger.info(f"Agent ready in room {LIVEKIT_ROOM}. Playground token: {playground_token}")
        await asyncio.Future()
    except Exception as e:
        logger.error(f"Failed to connect or start room: {e}")
        raise
    finally:
        _room_ready.clear()

async def main():
    global _http_session
    logger.info("Starting agent and web server...")
    _http_session = aiohttp.ClientSession()
    app = aiohttp.web.Application()
    app.router.add_get("/ui_websocket", ui_websocket)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", 5000)
    await site.start()
    logger.info("Web server ready at http://0.0.0.0:5000")
    livekit_task = asyncio.create_task(handle_room_connection())
    try:
        await livekit_task
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if _agent_room:
            await _agent_room.disconnect()
        if _http_session:
            await _http_session.close()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
