import asyncio
import logging
import os
import pandas as pd
import json
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession, RoomInputOptions, get_job_context
from livekit.plugins import deepgram, openai, silero, noise_cancellation, cartesia
from typing import Optional, List
import aiohttp
import redis.asyncio as redis
from faq_dataset import get_faq_data
from order_management import create_new_order, update_order
import difflib
import random
import requests
import smtplib
from email.mime.text import MIMEText
from fpdf import FPDF
from datetime import datetime
from email.message import EmailMessage
import re
import math
import time
from typing import Dict


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("LogisticsAgent")

AGENT_NAME = "inbound-agent"
LIVEKIT_ROOM_PREFIX = "call-"
ROOM_NAME = os.getenv("LIVEKIT_ROOM_NAME", f"{LIVEKIT_ROOM_PREFIX}-{random.randint(1000, 9999)}")

ORDERS_CSV_PATH = "orders.csv"
load_dotenv(".env")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIP_INBOUND_TRUNK_ID = os.getenv("SIP_INBOUND_TRUNK_ID")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = os.getenv("SMTP_PORT", 587)
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
RATES_TEAM_EMAIL = os.getenv("RATES_TEAM_EMAIL", "mahesh.v@qbotica.com")

FREIGHT_CLASS_MULTIPLIERS = {
    (0, 50): 0.80,
    (51, 85): 0.90,
    (86, 125): 1.15,
    (126, 175): 1.30,
    (176, 250): 1.50,
    (251, 500): 1.75
}

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, OPENAI_API_KEY, SIP_INBOUND_TRUNK_ID]):
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY":
        logger.warning("Google Maps API key not configured - LTL distance calculations will not work")
    logger.error("Missing required environment variables")
    raise ValueError("Please check .env for required API keys, LIVEKIT_URL, and SIP_INBOUND_TRUNK_ID")

if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, RATES_TEAM_EMAIL]):
    logger.warning("SMTP configuration incomplete - Email sending will not work")

try:
    logistics_dataset = pd.read_csv("orders.csv")
    required_columns = ["shipment_id", "origin_zip", "dest_zip", "weight_lbs", "freight_class",
                        "base_rate_per_mile_cwt", "fuel_surcharge_percent", "accessorial_charges",
                        "driver_id", "vendor_id", "skid_details", "is_stackable", "is_urgent",
                        "delivery_appointment", "origin_country", "dest_country", "status",
                        "created_at", "last_updated_timestamp"]
    if not all(col in logistics_dataset.columns for col in required_columns):
        logger.warning("Dataset missing some columns, initializing missing ones")
        for col in required_columns:
            if col not in logistics_dataset.columns:
                logistics_dataset[col] = None
        logistics_dataset.to_csv("orders.csv", index=False)
    logger.info(f"Loaded {len(logistics_dataset)} rows from dataset")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise



# def requires_verification(func):
#     async def wrapper(ctx: JobContext, *args, **kwargs):
#         cust_status = await cust_val(ctx)
#         if cust_status["status"] != "matched":
#             await session.generate_reply(instructions="Please verify your identity before proceeding.")
#             return {"error": "unauthorized"}
#         return await func(ctx, *args, **kwargs)
#     return wrapper

async def log_transcript(role: str, content: str):
    logger.info(f"[{role.upper()}]: {content}")




def calculate_backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    jitter = random.random() * base_delay
    backoff = min((2 ** attempt) * base_delay + jitter, max_delay)
    return backoff

async def tts_with_retry(session: AgentSession, text: str) -> bool:
    if not session or not session.tts or not session.vad:
        logger.warning("TTS/VAD not available")
        return False

    for attempt in range(3):
        try:
            vad_stream = session.vad.stream()
            tts_stream = await session.tts.synthesize(text)

            if hasattr(tts_stream, "__aiter__"):
                async for chunk in tts_stream:
                    vad_packet = await vad_stream.__anext__()
                    if vad_packet.is_speech:
                        logger.info("User speech detected - stopping TTS")
                        if hasattr(tts_stream, "aclose"):
                            await tts_stream.aclose()
                        return False
            return True

        except Exception as e:
            delay = calculate_backoff_with_jitter(attempt)
            logger.warning(f"TTS attempt {attempt + 1} failed: {str(e)}")
            await asyncio.sleep(delay)

    logger.error("TTS failed after 3 attempts")
    return False

async def process_and_respond(session: AgentSession, text: str):
    await log_transcript("assistant", text)

    if not await tts_with_retry(session, text):
        logger.error("All TTS attempts failed - falling back to silent mode")

def validate_postal_code(postal_code: str, country: str = "") -> bool:
    postal_code = postal_code.strip().upper()
    country = country.upper().strip()

    if country in ["CANADA"]:
        country = "CA"
    elif country in ["UNITED STATES", "US", "USA"]:
        country = "US"
    if not country:
        if re.match(r"^\d{5}(-\d{4})?$", postal_code):
            country = "US"
        elif re.match(r"^[A-Z]\d[A-Z][ ]?\d[A-Z]\d$", postal_code):
            country = "CA"

    if country == "US":
        return bool(re.match(r"^\d{5}(-\d{4})?$", postal_code))
    elif country == "CA":
        return bool(re.match(r"^[A-Z]\d[A-Z][ ]?\d[A-Z]\d$", postal_code))
    return False

def get_distance_miles(origin_zip: str, dest_zip: str, origin_country: str = "", dest_country: str = "") -> float:
    def format_postal(zip_code: str, country: str) -> str:
        zip_code = zip_code.strip().upper().replace(" ", "")
        if country == "":
            if re.match(r"^\d{5}(-\d{4})?$", zip_code):
                country = "US"
            elif re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", zip_code):
                country = "CA"
        if country == "CA" and len(zip_code) == 6:
            zip_code = f"{zip_code[:3]} {zip_code[3:]}"
        return f"{zip_code}, {country}"

    origin = format_postal(origin_zip, origin_country)
    dest = format_postal(dest_zip, dest_country)

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": dest,
        "units": "imperial",
        "key": GOOGLE_MAPS_API_KEY
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            distance_text = data["rows"][0]["elements"][0]["distance"]["text"]
            return float(distance_text.replace(",", "").split()[0])
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to get distance after {max_retries} attempts: {e}")

            delay = calculate_backoff_with_jitter(attempt)
            logger.warning(f"Distance API failed (attempt {attempt + 1}), retrying in {delay:.2f}s...")
            time.sleep(delay)

def get_class_multiplier(freight_class: int) -> float:
    for (low, high), multiplier in FREIGHT_CLASS_MULTIPLIERS.items():
        if low <= freight_class <= high:
            return multiplier
    raise ValueError("Invalid freight class")

def calculate_ltl_rate(
        origin_zip: str,
        dest_zip: str,
        weight_lbs: float,
        freight_class: int,
        base_rate_per_mile_cwt: float = 0.10,
        fuel_surcharge_percent: float = 20,
        accessorial_charges: float = 35,
        origin_country: str = "US",
        dest_country: str = "US"
) -> float:
    distance = get_distance_miles(origin_zip, dest_zip, origin_country, dest_country)
    cwt = weight_lbs / 100
    multiplier = get_class_multiplier(freight_class)
    base_cost = distance * base_rate_per_mile_cwt * cwt * multiplier
    fuel_surcharge = base_cost * (fuel_surcharge_percent / 100)
    total_cost = base_cost + fuel_surcharge + accessorial_charges
    return round(total_cost, 2)

def parse_user_query(query: str) -> List[str]:
    query = query.lower().strip()
    field_keywords = {
        "shipment_id": ["shipment id", "id", "order number"],
        "shipment_id": ["origin id", "id"],
        "origin_zip": ["origin", "from", "origin", "starting", "pickup"],
        "dest_zip": ["destination", "to", "delivery"],
        "weight_lbs": ["weight", "pounds", "lbs"],
        "freight_class": ["freight class", "class"],
        "driver_id": ["driver", "driver id"],
        "vendor_id": ["vendor", "vendor id"],
        "skid_details": ["skid", "pallets", "skids"],
        "is_stackable": ["stackable", "stack"],
        "is_urgent": ["urgent", "hot", "rush"],
        "delivery_appointment": ["appointment", "delivery time"],
        "quote_amount": ["quote", "cost", "price", "rate"],
        "origin_country": ["origin country", "from country"],
        "dest_country": ["destination country", "to country"],
        "status": ["status", "track", "tracking"]
    }
    requested_fields = []
    for field, keywords in field_keywords.items():
        if any(keyword in query for keyword in keywords):
            requested_fields.append(field)
    return requested_fields

class LogisticsAgent(Agent):
    def __init__(self, room: rtc.Room):
        super().__init__(
            instructions=(
                f"You are calling from Polaris, start the convo with 'Thank you for calling Polaris. How may I assist you?' and don't let them know you are an AI agent. "
                "Always verify them before talking any business using 'cust_val' tool. and also tell their company name with the name "
                "if they want to create,track,update  a shipment always verify them first using 'cust_val' tool and then ask for the shipment ID."
                "if they are not verified end the call politely with 'hangup' tool."
                "Use more pausewords like 'uhmm.' and 'Hmm...' to sound more human."
                "Be precise and act and talk like karen, confirming details like company name and shipment specifics."
                "Only use 'transfer_call' when explicitly requested by the user or after 3 failed attempts to resolve their issue."
                "Ask the details of the shipment one by one when an user is creating an order"
                "And after creating the order tell them the price of the shipment first and then ask them to note the shipment ID. And be ready for the followup questions instead of ending the call."
                "After calculating the rate just tell them and confirm it before creating the order"
                "if the shipment is InTransit reply it cleared the customs and will receive soon"
                "If they wanna ship hazardous goods tell them you cant ship and ask them to reach out to Shipping department."
                "There ight be a mix up in the words sometime so you need to ask them for clarification politely and do not cut the call prematurely. "
                "Try to end the call for only persons who is legitly wsting your time. Dont try to end the call for persons who is creating a valid order or asking a valid question."
                "Do not repeat the shipment ID in follow-up questions. Say 'Uhh... let me quickly check that shipment ID number for you, okay? Just a sec...' when checking details of."
                "Fact-check shipment details using common sense and cut the call immediatly using 'hangup' (e.g., reject shipments to invalid locations like Antarctica , ship to other planets)."
                "If the user is wasting your time, be patient 1 time or 2 times, then end the call gracefully with the 'end_call' tool."
                "Use the 'get_quote' tool when the user provides a valid shipment ID number (e.g., SHP12345) or complete origin/destination ZIP code or codes (e.g., 10001 to L9T5A6)."
                "For follow-up questions about a quote, quote about use, a latest quote from context and respond only with the specific details requested (e.g., about weight, stackability) without repeating the full quote about."
                "Use the 'calculate_ltl_quote' tool for precise shipping LTL rates when the user provides ZIP code , weight rate, freight class rate, and skid details about."
                "If the order has more than 7 items skids, escalate to rates team by collecting email from customer email and using a 'send email' tool to notify rates team."
                "Ask for skid dimensions (length,width and height in inches), stackability (stackable or non-stackable), delivery appointment requirements, and urgency (e.g., same-day pickup)."
                "Use the 'create_new_order' tool to create a new shipping order when the user wants to send a load. Ask for origin country and state and city and zip code and destination country and state and city and zip code , weight (in pounds), freight class, skid details, stackability, and delivery requirements."
                "Assign a unique shipment ID starting with 'SHP' followed by random numbers (e.g., SHP12345)."
                "You wanna tell the price and confirm them to create a shipent"
                "Use the 'update_order' tool to update an existing order when the user provides a shipment ID and at least one field to update. Retain existing details for unchanged fields."
                "If the update request is unclear, ask for clarification (e.g., 'Could you confirm which fields you'd like to update for Shipment ID {shipment_id}?')."
                "Use the 'get_faq' tool for general questions about logistics services, including driver jobs, complaints, or how to ship."
                "If the user asks for a joke, tell one."
                "The 'get_quote', 'create_new_order', and 'update_order' tools automatically send shipment details to a Slack channel. Do not call 'send_slack_message' directly."
                "If the user asks for a human agent, use 'transfer_call' to connect them"
                "If a message is received from Slack, incorporate it appropriately."
                "After an order is created, if asked about status, say it will be processed soon."
                "For urgent orders (e.g., same-day pickup), mark as 'hot' and escalate via email or transfer. to '+16235009891'"
                "Confirm details with phrases like 'Let me make sure I have that right' "
                "Shipment Id always starts with SHP followed by 5 digits."

            )
        )
        self.call_context = {
            'call_type': 'waiting',
            'outbound_shipment_context': {},
            'conversation_history': [],
            'latest_quote': None,
            'call_active': False,
            'latest_query': None

        }
        self.room = room
        self.api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        self.room.on("data_received", lambda packet: self.on_data_received(packet))

    @agents.function_tool(name='cust_val', description="Validate the caller from HubSpot CRM using their phone number.")
    async def cust_val(ctx: JobContext) -> Dict:
        try:
            phone_number = ctx.room.name.split('_')[1]
            logger.info(f"Verifying phone number: {phone_number}")

            from hubby import search_contact_by_phone
            result = search_contact_by_phone(phone_number)

            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            await redis_client.set(f"verified:{phone_number}", "true" if result["status"] == "success" else "false")
            await log_transcript("assistant", result["message"])

            return result

        except Exception as e:
            logger.error(f"Error verifying customer: {e}")
            response = "Sorry, I couldn't verify your identity. Please try again."
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            await redis_client.set(f"verified:{phone_number}", "false")
            await log_transcript("assistant", response)
            return {
                "status": "error",
                "contact": None,
                "message": response
            }


    def requires_verification(self):
        """Check if customer is verified before proceeding with sensitive operations"""
        return self.call_context.get('customer_verified', False)

    async def cleanup(self):
        try:
            await self.api.aclose()
            logger.info("LiveKit API client closed")
        except Exception as e:
            logger.error(f"Error closing LiveKit API client: {e}")

    def on_data_received(self, packet: rtc.DataPacket):
        try:
            payload = packet.data.decode('utf-8')
            logger.debug(f"Received data packet: {payload}")
            data = json.loads(payload)
            slack_message = data.get("slack_message")
            if slack_message:
                logger.info(f"Received Slack message: {slack_message}")
                response = f"I received a message from our team via Slack: {slack_message}. How can I assist you further?"
                asyncio.create_task(process_and_respond(session, response))
            else:
                logger.warning(f"Invalid data packet: {payload}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse data packet: {payload}")
        except Exception as e:
            logger.error(f"Error processing data packet: {e}")

    @agents.function_tool(name="send_email",
                          description="Send an email with shipment details to the customer and rates team.")
    async def send_email(self, ctx: agents.RunContext, to_email: str, subject: str, body: str) -> dict:
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = SMTP_USERNAME
            msg['To'] = to_email
            msg['Cc'] = RATES_TEAM_EMAIL

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(SMTP_USERNAME, [to_email, RATES_TEAM_EMAIL], msg.as_string())

            logger.info(f"Email sent to {to_email} with CC to {RATES_TEAM_EMAIL}")
            return {"status": "success", "response": "Email sent successfully."}
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {"status": "failed", "response": f"Failed to send email: {str(e)}"}

    @agents.function_tool(
        name='send_pod',
        description="Send a proof of delivery after a user created an order and also if they update any order"
    )
    async def send_pod_mail(self, ctx: agents.RunContext, to_email: str, subject: str, body: str) -> dict:
        try:
            match = re.search(r"SHP\d{5}", body)
            if not match:
                return {"status": "failed", "response": "Shipment ID not found in email body."}

            shipment_id = match.group(0)
            csv_file = "orders.csv"
            logo_file = "test.png"
            output_pdf = f"order_summary_{shipment_id}.pdf"
            selected_columns = ["shipment_id", "origin_zip", "dest_zip", "Quote"]

            df = pd.read_csv(csv_file)
            df = df[df["shipment_id"] == shipment_id]

            if df.empty:
                return {"status": "failed", "response": f"Shipment ID {shipment_id} not found in CSV."}

            df = df[selected_columns]
            today = datetime.today().strftime('%Y-%m-%d')

            pdf = FPDF()
            pdf.add_page()

            if os.path.exists(logo_file):
                pdf.image(logo_file, x=10, y=10, w=40)
                pdf.set_xy(10, 40)
            else:
                pdf.set_y(10)

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Your Order Summary", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Date: {today}", ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Shipment Details for ID: {shipment_id}", ln=True)

            pdf.set_font("Arial", "B", 11)
            col_width = 45
            for col in selected_columns:
                pdf.cell(col_width, 10, col, border=1)
            pdf.ln()

            pdf.set_font("Arial", "", 11)
            for _, row in df.iterrows():
                for col in selected_columns:
                    pdf.cell(col_width, 10, str(row[col]), border=1)
                pdf.ln()

            pdf.ln(10)
            pdf.set_font("Arial", "I", 10)
            pdf.multi_cell(0, 8,
                           "Your Shipment has been successfully booked.\nFor any queries regarding this shipment, contact: support@qbotica.com")

            pdf.output(output_pdf)

            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = SMTP_USERNAME
            msg['To'] = to_email
            msg['Cc'] = RATES_TEAM_EMAIL
            msg.set_content(body)

            with open(output_pdf, 'rb') as f:
                msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename=output_pdf)

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)

            logger.info(f"POD email sent to {to_email} with PDF for {shipment_id}.")
            return {"status": "success", "response": f"POD email sent for {shipment_id}."}

        except Exception as e:
            logger.error(f"Error sending POD email: {e}")
            return {"status": "failed", "response": f"Failed to send POD email: {str(e)}"}

    @agents.function_tool(name="calculate_ltl_quote",
                          description="Calculate LTL shipping rate using origin/destination ZIP codes, weight, freight class, and skid details.")
    async def calculate_ltl_quote_tool(
            self,
            ctx: agents.RunContext,
            origin_zip: str,
            dest_zip: str,
            weight_lbs: float,
            freight_class: int = 125,
            base_rate_per_mile_cwt: float = 0.10,
            fuel_surcharge_percent: float = 20,
            accessorial_charges: float = 35,
            skid_details: Optional[List[Dict[str, float]]] = None,
            is_stackable: bool = True,
            is_urgent: bool = False,
            delivery_appointment: bool = False,
            customer_email: Optional[str] = None,
            origin_country: str = "US",
            dest_country: str = "US"
    ) -> dict:
        try:
            if not validate_postal_code(origin_zip, origin_country):
                raise ValueError(f"Invalid origin postal code: {origin_zip}")
            if not validate_postal_code(dest_zip, dest_country):
                raise ValueError(f"Invalid destination postal code: {dest_zip}")

            skid_count = len(skid_details) if skid_details else 1
            if skid_count > 7:
                if not customer_email:
                    response = "For orders with more than 7 skids, I need your email address to escalate to our rates team."
                    return {"response": response, "rate": None, "distance": None}

                skid_summary = "\n".join(
                    [f"Skid {i + 1}: {d['length']}x{d['width']}x{d['height']} inches, {d['weight']} lbs"
                     for i, d in enumerate(skid_details)]) if skid_details else "No skid details provided"
                email_body = (
                    f"Urgent Quote Request{' (Hot)' if is_urgent else ''}:\n"
                    f"From: {origin_zip} ({origin_country})\n"
                    f"To: {dest_zip} ({dest_country})\n"
                    f"Weight: {weight_lbs} lbs\n"
                    f"Freight Class: {freight_class}\n"
                    f"Skids: {skid_count}\n"
                    f"Skid Details:\n{skid_summary}\n"
                    f"Stackable: {'Yes' if is_stackable else 'No'}\n"
                    f"Delivery Appointment: {'Required' if delivery_appointment else 'Not required'}\n"
                    f"Customer Email: {customer_email}"
                )
                email_result = await self.send_email(
                    ctx,
                    to_email=customer_email,
                    subject=f"Urgent Quote Request: {skid_count} Skids from {origin_zip} to {dest_zip}{' (Hot)' if is_urgent else ''}",
                    body=email_body
                )
                response = (
                    f"For {skid_count} skids, I've sent an email to {customer_email} and copied our rates team. "
                    f"They'll get back to you soon with a quote{' marked as urgent' if is_urgent else ''}. "
                    "Anything else I can help with?"
                )
                return {"response": response, "rate": None, "distance": None}

            rate = calculate_ltl_rate(
                origin_zip=origin_zip,
                dest_zip=dest_zip,
                weight_lbs=weight_lbs,
                freight_class=freight_class,
                base_rate_per_mile_cwt=base_rate_per_mile_cwt,
                fuel_surcharge_percent=fuel_surcharge_percent,
                accessorial_charges=accessorial_charges,
                origin_country=origin_country,
                dest_country=dest_country
            )
            distance = get_distance_miles(origin_zip, dest_zip, origin_country, dest_country)
            skid_summary = f", {skid_count} skids" if skid_count > 1 else ""
            response = (
                f"Dont use Asterisk LTL shipping quote from {origin_zip} ({origin_country}) to {dest_zip} ({dest_country}): "
                f"${rate:.2f} for {weight_lbs} lbs (freight class {freight_class}{skid_summary}). "
                f"Distance: {distance:.1f} miles. "
                f"Stackable: {'Yes' if is_stackable else 'No'}. "
                f"Delivery: {'Appointment required' if delivery_appointment else 'No appointment'}. "
                f"{'Marked as urgent.' if is_urgent else ''}"
                f"This includes base rate, fuel surcharge ({fuel_surcharge_percent}%), and accessorial charges (${accessorial_charges})."
            )
            slack_message = (
                f"LTL quote calculated: ${rate:.2f} from {origin_zip} ({origin_country}) to {dest_zip} ({dest_country}), "
                f"Weight: {weight_lbs} lbs, Class: {freight_class}, Skids: {skid_count}, Distance: {distance:.1f} mi, "
                f"Stackable: {'Yes' if is_stackable else 'No'}, Urgent: {'Yes' if is_urgent else 'No'}"
            )
            await self.send_slack_message(ctx, message=slack_message)
            return {"response": response, "rate": rate, "distance": distance}

        except Exception as e:
            logger.error(f"Error calculating LTL rate: {e}")
            response = "I apologize, but I'm having trouble calculating that LTL rate right now. Please verify the postal codes and try again."
            return {"response": response, "rate": None, "distance": None}

    @agents.function_tool(name="get_quote",
                          description="Retrieve specific shipment details based on shipment ID or origin/destination postal codes.")
    async def get_quote(
            self,
            ctx: agents.RunContext,
            shipment_id: Optional[str] = None,
            origin_zip: Optional[str] = None,
            dest_zip: Optional[str] = None,
            weight_lbs: Optional[float] = None,
            freight_class: Optional[int] = None,
            driver_id: Optional[str] = None,
            vendor_id: Optional[str] = None,
            origin_country: str = "US",
            dest_country: str = "US",
            user_query: Optional[str] = None
    ) -> dict:
        try:
            if user_query and self.call_context['latest_quote'] and not (shipment_id or origin_zip or dest_zip):
                requested_fields = parse_user_query(user_query)
                if requested_fields:
                    shipment = self.call_context['latest_quote']['shipment_data']
                    quote_amount = self.call_context['latest_quote']['quote_amount']
                    response_parts = []
                    for field in requested_fields:
                        if field == "quote_amount" and quote_amount is not None:
                            response_parts.append(f"The quote is ${quote_amount:.2f}.")
                        elif shipment.get(field) is not None and not isinstance(shipment[field], float):
                            if field == "is_stackable":
                                response_parts.append(f"Stackable: {'Yes' if shipment[field] else 'No'}.")
                            elif field == "is_urgent":
                                response_parts.append(f"{'Marked as urgent.' if shipment[field] else 'Not urgent.'}")
                            elif field == "delivery_appointment":
                                response_parts.append(
                                    f"Delivery: {'Appointment required' if shipment[field] else 'No appointment'}.")
                            elif field == "skid_details" and shipment[field]:
                                skid_details = json.loads(shipment[field]) if isinstance(shipment[field], str) else \
                                shipment[field]
                                response_parts.append(f"Skids: {len(skid_details)}.")
                            else:
                                response_parts.append(f"{field.replace('_', ' ').title()}: {shipment[field]}.")
                    if response_parts:
                        response = "Let me make sure I have that right. " + " ".join(response_parts)
                        await log_transcript("assistant", response)
                        self.call_context['conversation_history'].append({"query": user_query, "response": response})
                        return {"response": response}
                    else:
                        response = "I couldn't find those specific details in the latest quote. Could you clarify what you need?"
                        await log_transcript("assistant", response)
                        return {"response": response}

            quote_info = get_quote_from_dataset({
                "shipment_id": shipment_id,
                "origin_zip": origin_zip,
                "dest_zip": dest_zip,
                "weight_lbs": weight_lbs,
                "freight_class": freight_class,
                "driver_id": driver_id,
                "vendor_id": vendor_id,
                "origin_country": origin_country,
                "dest_country": dest_country
            })
            self.call_context['latest_quote'] = quote_info
            logger.info(f"Quote retrieved: {quote_info}")

            if quote_info["quote_amount"] is not None:
                shipment = quote_info["shipment_data"]
                requested_fields = parse_user_query(user_query) if user_query else ["shipment_id", "origin_zip",
                                                                                    "dest_zip", "quote_amount"]
                response_parts = ["Let me make sure I have that right."]
                for field in requested_fields:
                    if field == "quote_amount":
                        response_parts.append(
                            f"The quote for shipment {shipment['shipment_id']} is ${quote_info['quote_amount']:.2f}.")
                    elif shipment.get(field) is not None and not isinstance(shipment[field], float):
                        if field == "is_stackable":
                            response_parts.append(f"Stackable: {'Yes' if shipment[field] else 'No'}.")
                        elif field == "is_urgent":
                            response_parts.append(f"{'Marked as urgent.' if shipment[field] else 'Not urgent.'}")
                        elif field == "delivery_appointment":
                            response_parts.append(
                                f"Delivery: {'Appointment required' if shipment[field] else 'No appointment'}.")
                        elif field == "skid_details" and shipment[field]:
                            skid_details = json.loads(shipment[field]) if isinstance(shipment[field], str) else \
                            shipment[field]
                            response_parts.append(f"Skids: {len(skid_details)}.")
                        else:
                            response_parts.append(f"{field.replace('_', ' ').title()}: {shipment[field]}.")
                response = " ".join(response_parts)
                slack_message = (
                    f"Shipment query: ID {shipment['shipment_id']} from {shipment['origin_zip']} "
                    f"to {shipment['dest_zip']}. Quote: ${quote_info['quote_amount']:.2f}, "
                    f"Weight: {shipment['weight_lbs']} lbs, Freight Class: {shipment['freight_class']}."
                )
                await self.send_slack_message(ctx, message=slack_message)
                self.call_context['conversation_history'].append(
                    {"query": user_query or "get_quote", "response": response})
                return {"response": response}
            else:
                if quote_info["shipment_data"]:
                    shipment = quote_info["shipment_data"]
                    response = (
                        f"I found Shipment ID {shipment['shipment_id']} from {shipment['origin_zip']} "
                        f"to {shipment['dest_zip']} with a weight of {shipment['weight_lbs']} lbs. "
                        f"However, I couldn't calculate a quote. Would you like to provide additional details or proceed with another request?"
                    )
                else:
                    response = "No quote found for the provided details. Could you verify the shipment ID or provide origin and destination postal codes?"
                self.call_context['conversation_history'].append(
                    {"query": user_query or "get_quote", "response": response})
                return {"response": response}

        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            response = "I apologize, but I'm having trouble retrieving that quote right now. Please try again."
            self.call_context['conversation_history'].append({"query": user_query or "get_quote", "response": response})
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

            return {"response": response}

        except Exception as e:
            logger.error(f"Error getting FAQ answer: {e}")
            response = "I apologize, but I'm having trouble answering that question right now. Please try again."
            return {"response": response}

    @agents.function_tool(name="create_new_order", description="Create a new shipping order with provided details.")
    async def create_new_order_tool(
            self,
            ctx: agents.RunContext,
            origin_zip: str,
            dest_zip: str,
            weight_lbs: float,
            freight_class: int = 125,
            base_rate_per_mile_cwt: float = 0.10,
            fuel_surcharge_percent: float = 20,
            accessorial_charges: float = 35,
            driver_id: Optional[str] = None,
            vendor_id: Optional[str] = None,
            skid_details: Optional[List[Dict[str, float]]] = None,
            is_stackable: bool = True,
            is_urgent: bool = False,
            delivery_appointment: bool = False,
            customer_email: Optional[str] = None,
            origin_country: str = "US",
            dest_country: str = "US",
            Quote: float = None
    ) -> dict:


        try:
            if not validate_postal_code(origin_zip, origin_country):
                raise ValueError(f"Invalid origin postal code: {origin_zip}")
            if not validate_postal_code(dest_zip, dest_country):
                raise ValueError(f"Invalid destination postal code: {dest_zip}")

            skid_count = len(skid_details) if skid_details else 1
            if skid_count > 7:
                if not customer_email:
                    response = "For orders with more than 7 skids, I need your email address to escalate to our rates team."
                    await log_transcript("assistant", response)
                    return {"status": "failed", "response": response}

                skid_summary = "\n".join(
                    [f"Skid {i + 1}: {d['length']}x{d['width']}x{d['height']} inches, {d['weight']} lbs"
                     for i, d in enumerate(skid_details)]) if skid_details else "No skid details provided"
                email_body = (
                    f"Urgent Order Request{' (Hot)' if is_urgent else ''}:\n"
                    f"From: {origin_zip} ({origin_country})\n"
                    f"To: {dest_zip} ({dest_country})\n"
                    f"Weight: {weight_lbs} lbs\n"
                    f"Freight Class: {freight_class}\n"
                    f"Skids: {skid_count}\n"
                    f"Skid Details:\n{skid_summary}\n"
                    f"Stackable: {'Yes' if is_stackable else 'No'}\n"
                    f"Delivery Appointment: {'Required' if delivery_appointment else 'Not required'}\n"
                    f"Customer Email: {customer_email}"
                )
                email_result = await self.send_email(
                    ctx,
                    to_email=customer_email,
                    subject=f"Urgent Order Request: {skid_count} Skids from {origin_zip} to {dest_zip}{' (Hot)' if is_urgent else ''}",
                    body=email_body
                )
                response = (
                    f"For {skid_count} skids, I've sent an email to {customer_email} and copied our rates team. "
                    f"They'll process your order soon{' marked as urgent' if is_urgent else ''}. "
                    "Anything else I can help with?"
                )
                await log_transcript("assistant", response)
                return {"status": "success", "response": response}

            result = create_new_order(
                origin_zip=origin_zip,
                dest_zip=dest_zip,
                weight_lbs=weight_lbs,
                freight_class=freight_class,
                base_rate_per_mile_cwt=base_rate_per_mile_cwt,
                fuel_surcharge_percent=fuel_surcharge_percent,
                accessorial_charges=accessorial_charges,
                driver_id=driver_id,
                vendor_id=vendor_id,
                skid_details=skid_details,
                is_stackable=is_stackable,
                is_urgent=is_urgent,
                delivery_appointment=delivery_appointment,
                origin_country=origin_country,
                dest_country=dest_country,
                Quote=Quote
            )
            response = result["response"]
            response = session.generate_reply(instructions=f"Return coherent response for {response}")
            if result["status"] == "success":
                skid_count = len(skid_details) if skid_details else 1
                slack_message = (
                    f"New order created: {response} "
                    f"Details: From {origin_zip} ({origin_country}) to {dest_zip} ({dest_country}), "
                    f"Weight: {weight_lbs} lbs, Freight Class: {freight_class}, Skids: {skid_count}, "
                    f"Stackable: {'Yes' if is_stackable else 'No'}, Urgent: {'Yes' if is_urgent else 'No'}"
                )
                await self.send_slack_message(ctx, message=slack_message)
            return result
        except Exception as e:
            logger.error(f"Error creating new order: {e}")
            response = "I apologize, but I'm having trouble creating that order right now. Please try again."
            await log_transcript("assistant", response)
            return {"status": "failed", "response": response}

    @agents.function_tool(name="update_order", description="Update an existing shipping order with provided details.")
    async def update_order_tool(
            self,
            ctx: agents.RunContext,
            shipment_id: str,
            origin_zip: Optional[str] = None,
            dest_zip: Optional[str] = None,
            weight_lbs: Optional[float] = None,
            freight_class: Optional[int] = None,
            base_rate_per_mile_cwt: Optional[float] = None,
            fuel_surcharge_percent: Optional[float] = None,
            accessorial_charges: Optional[float] = None,
            driver_id: Optional[str] = None,
            vendor_id: Optional[str] = None,
            skid_details: Optional[List[Dict[str, float]]] = None,
            is_stackable: Optional[bool] = None,
            is_urgent: Optional[bool] = None,
            delivery_appointment: Optional[bool] = None,
            origin_country: Optional[str] = None,
            dest_country: Optional[str] = None
    ) -> dict:

        try:
            kwargs = {
                k: v for k, v in {
                    "origin_zip": origin_zip,
                    "dest_zip": dest_zip,
                    "weight_lbs": weight_lbs,
                    "freight_class": freight_class,
                    "base_rate_per_mile_cwt": base_rate_per_mile_cwt,
                    "fuel_surcharge_percent": fuel_surcharge_percent,
                    "accessorial_charges": accessorial_charges,
                    "driver_id": driver_id,
                    "vendor_id": vendor_id,
                    "skid_details": skid_details,
                    "is_stackable": is_stackable,
                    "is_urgent": is_urgent,
                    "delivery_appointment": delivery_appointment,
                    "origin_country": origin_country,
                    "dest_country": dest_country
                }.items() if v is not None
            }
            result = update_order(shipment_id=shipment_id, **kwargs)
            response = result["response"]
            response = session.generate_reply(instructions=f"Return coherent response for {response}")
            if result["status"] == "success":
                updated_fields = {k: v for k, v in kwargs.items()}
                slack_message = (
                    f"Order updated: {response} "
                    f"Shipment ID: {shipment_id}, Updated fields: {', '.join(f'{k}: {v}' for k, v in updated_fields.items())}"
                )
                await self.send_slack_message(ctx, message=slack_message)
            return result
        except Exception as e:
            logger.error(f"Error updating order: {e}")
            response = f"Could you confirm which fields you'd like to update for Shipment ID {shipment_id}?"
            await log_transcript("assistant", response)
            return {"status": "failed", "response": response}

    @agents.function_tool(name="transfer_call", description="Transfer the call to a human agent.")
    async def transfer_call(self, ctx: agents.RunContext, transfer_to: Optional[str] = None) -> dict:
        transfer_to = transfer_to or os.getenv("TRANSFER_PHONE_NUMBER", "+16235009891")
        if not transfer_to:
            logger.error("No transfer phone number configured")
            response = "I'm sorry, I cannot transfer the call at this time. Please try again later."
            await log_transcript("assistant", response)
            return {"response": response}

        logger.info(f"Initiating call transfer to {transfer_to}")

        try:
            response = "Please hold while I transfer you to a human agent."
            await session.generate_reply(instructions=response)

            await asyncio.sleep(2)

            participants = await self.api.room.list_participants(
                api.ListParticipantsRequest(room=self.room.name)
            )
            if not participants or not participants.participants:
                logger.error("No participants found in the room")
                response = "I'm sorry, I couldn't find an active caller to transfer. How else can I assist you?"
                await log_transcript("assistant", response)
                return {"response": response}

            for p in participants.participants:
                logger.debug(
                    f"Participant: identity={p.identity}, metadata={p.metadata}, attributes={p.attributes}, state={p.state}")

            participant = next(
                (p for p in participants.participants
                 if p.identity != AGENT_NAME and
                 (p.attributes.get("sip.callID") or p.attributes.get("sip.callStatus") == "active")),
                None
            )
            if not participant:
                logger.error("No valid SIP participant found for transfer")
                response = (
                    "I'm sorry, I couldn't connect you to a human agent because the call isn't set up for transfer. "
                    "Please try again or let me assist you with your shipping needs."
                )
                await log_transcript("assistant", response)
                return {"response": response}

            logger.info(f"Selected participant for transfer: identity={participant.identity}, "
                        f"sip.callID={participant.attributes.get('sip.callID')}, "
                        f"sip.callStatus={participant.attributes.get('sip.callStatus')}")

            await self.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self.room.name,
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
            response = "Thank you for calling Polaris logistics. Have a great day! You take care!"
            await log_transcript("assistant", response)

            await asyncio.sleep(2)
            await self.hangup()
            await self.cleanup()
            return {"response": response}

        except Exception as e:
            logger.error(f"Error ending call: {e}")
            response = "Goodbye! You take care!"
            await log_transcript("assistant", response)
            await self.cleanup()
            return {"response": response}

    async def hangup(self):
        response = "Tell you are wasting my time rudely"
        await session.generate_reply(instructions=response)
        await asyncio.sleep(0.5)
        await self.api.room.delete_room(
            api.DeleteRoomRequest(
                room=self.room.name,
            )
        )

def get_quote_from_dataset(parsed_entities: dict) -> dict:
    if not parsed_entities:
        return {"quote_amount": None, "shipment_data": None, "response": "No shipment details provided."}

    shipment = None
    shipment_id = parsed_entities.get("shipment_id")
    origin_zip = parsed_entities.get("origin_zip")
    dest_zip = parsed_entities.get("dest_zip")
    origin_country = parsed_entities.get("origin_country", "US")
    dest_country = parsed_entities.get("dest_country", "US")

    def clean_dict(d: dict) -> dict:
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned

    if os.path.exists(ORDERS_CSV_PATH):
        orders_df = pd.read_csv(ORDERS_CSV_PATH)
        numeric_columns = ["weight_lbs", "freight_class", "base_rate_per_mile_cwt",
                           "fuel_surcharge_percent", "accessorial_charges"]
        for col in numeric_columns:
            if col in orders_df.columns:
                orders_df[col] = pd.to_numeric(orders_df[col], errors="coerce")

        if shipment_id:
            shipment_id = shipment_id.upper()
            matching_shipments = orders_df[orders_df["shipment_id"].str.upper() == shipment_id]
            if not matching_shipments.empty:
                shipment = matching_shipments.iloc[0]
                try:
                    quote = calculate_ltl_rate(
                        origin_zip=str(shipment['origin_zip']),
                        dest_zip=str(shipment['dest_zip']),
                        weight_lbs=float(shipment['weight_lbs']),
                        freight_class=int(shipment['freight_class']),
                        base_rate_per_mile_cwt=float(shipment.get('base_rate_per_mile_cwt', 0.10)),
                        fuel_surcharge_percent=float(shipment.get('fuel_surcharge_percent', 20)),
                        accessorial_charges=float(shipment.get('accessorial_charges', 35)),
                        origin_country=shipment.get('origin_country', 'US'),
                        dest_country=shipment.get('dest_country', 'US')
                    )
                    return {
                        "quote_amount": round(quote, 2),
                        "shipment_data": clean_dict(shipment.to_dict()),
                        "response": "Quote retrieved successfully."
                    }
                except Exception as e:
                    logger.warning(f"LTL calculation failed, using fallback: {e}")
                    quote = float(shipment["weight_lbs"]) * 0.05
                    return {
                        "quote_amount": round(quote, 2),
                        "shipment_data": clean_dict(shipment.to_dict()),
                        "response": "Quote retrieved using fallback calculation."
                    }

    if shipment_id:
        matching_shipments = logistics_dataset[logistics_dataset["shipment_id"].str.upper() == shipment_id.upper()]
        if not matching_shipments.empty:
            shipment = matching_shipments.iloc[0]
            try:
                quote = calculate_ltl_rate(
                    origin_zip=str(shipment['origin_zip']),
                    dest_zip=str(shipment['dest_zip']),
                    weight_lbs=float(shipment['weight_lbs']),
                    freight_class=int(shipment['freight_class']),
                    base_rate_per_mile_cwt=float(shipment.get('base_rate_per_mile_cwt', 0.10)),
                    fuel_surcharge_percent=float(shipment.get('fuel_surcharge_percent', 20)),
                    accessorial_charges=float(shipment.get('accessorial_charges', 35)),
                    origin_country=shipment.get('origin_country', 'US'),
                    dest_country=shipment.get('dest_country', 'US')
                )
                return {
                    "quote_amount": round(quote, 2),
                    "shipment_data": clean_dict(shipment.to_dict()),
                    "response": "Quote retrieved successfully."
                }
            except Exception as e:
                logger.warning(f"LTL calculation failed for dataset, using fallback: {e}")
                quote = float(shipment["weight_lbs"]) * 0.05
                return {
                    "quote_amount": round(quote, 2),
                    "shipment_data": clean_dict(shipment.to_dict()),
                    "response": "Quote retrieved using fallback calculation."
                }

    if origin_zip and dest_zip and parsed_entities.get("weight_lbs") and parsed_entities.get("freight_class"):
        try:
            quote = calculate_ltl_rate(
                origin_zip=origin_zip,
                dest_zip=dest_zip,
                weight_lbs=float(parsed_entities["weight_lbs"]),
                freight_class=int(parsed_entities["freight_class"]),
                base_rate_per_mile_cwt=float(parsed_entities.get("base_rate_per_mile_cwt", 0.10)),
                fuel_surcharge_percent=float(parsed_entities.get("fuel_surcharge_percent", 20)),
                accessorial_charges=float(parsed_entities.get("accessorial_charges", 35)),
                origin_country=origin_country,
                dest_country=dest_country
            )
            shipment_data = {
                "shipment_id": f"SHP{random.randint(10000, 99999)}",
                "origin_zip": origin_zip,
                "dest_zip": dest_zip,
                "weight_lbs": parsed_entities["weight_lbs"],
                "freight_class": parsed_entities["freight_class"],
                "base_rate_per_mile_cwt": parsed_entities.get("base_rate_per_mile_cwt", 0.10),
                "fuel_surcharge_percent": parsed_entities.get("fuel_surcharge_percent", 20),
                "accessorial_charges": parsed_entities.get("accessorial_charges", 35),
                "driver_id": parsed_entities.get("driver_id"),
                "vendor_id": parsed_entities.get("vendor_id"),
                "origin_country": origin_country,
                "dest_country": dest_country
            }
            return {
                "quote_amount": round(quote, 2),
                "shipment_data": clean_dict(shipment_data),
                "response": "Quote retrieved successfully."
            }
        except Exception as e:
            logger.error(f"Error calculating quote with postal codes: {e}")
            return {
                "quote_amount": None,
                "shipment_data": None,
                "response": "Failed to calculate quote with provided postal codes. Please verify details."
            }

    return {
        "quote_amount": None,
        "shipment_data": None,
        "response": f"Shipment ID {shipment_id} not found in orders or dataset, and insufficient details provided."
    }


def get_faq_answer(question: str) -> Optional[str]:
    question = question.lower().strip()
    faq_data = get_faq_data()
    questions = list(faq_data.keys())
    matches = difflib.get_close_matches(question, questions, n=1, cutoff=0.6)
    if matches:
        return faq_data[matches[0]]
    return None

async def health_check():
    """Silent health monitor with failure alerts"""
    while True:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{LIVEKIT_URL}/health") as resp:
                    if resp.status != 200:
                        logger.error(f"Health check failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"Health check error: {e}")

        await asyncio.sleep(30)

async def entrypoint(ctx: JobContext):
    health_check_task = None
    global session
    session = None
    connection_active = True
    inactivity_timeout = 300
    last_activity_time = asyncio.get_event_loop().time()



    try:
        logger.info(f"Starting {AGENT_NAME} entrypoint for room: {ctx.room.name}")

        try:
            phone_number = ctx.room.name.split('_')[1]
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            await redis_client.set(f"room:{phone_number}", ctx.room.name)
            logger.info(f"Added room {ctx.room.name} to Redis for phone: {phone_number}")
        except Exception as e:
            logger.error(f"Error adding phone number to Redis: {e}")

        await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)
        logger.info(f"Agent connected to {LIVEKIT_URL}, waiting for participant...")


        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(),
                timeout=60.0
            )
            logger.info(
                f"Participant joined: {participant.identity}, kind: {participant.kind}, attributes: {participant.attributes.items()}")
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for participant")
            return False
        except Exception as e:
            logger.error(f"Error waiting for participant: {e}")
            return False

        agent = LogisticsAgent(ctx.room)
        session = AgentSession(
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o", temperature=0.7),
            tts=cartesia.TTS(voice="2e926772-e6a4-4cdf-85bf-368105e8e424"),
            vad=silero.VAD.load()
        )

        agent.call_context['call_type'] = 'inbound'
        agent.call_context['conversation_history'] = []
        agent.call_context['latest_quote'] = None
        agent.call_context['call_active'] = True

        async def on_transcription(transcription: str):
            if transcription:
                agent.call_context['latest_query'] = transcription
                logger.debug(f"User query captured: {transcription}")

        health_check_task = asyncio.create_task(health_check())
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
        )

        welcome_message = "Thank you for calling Logistics agent. How may I assist you?"
        await session.generate_reply(instructions=welcome_message)

        logger.info("Inbound call initiated successfully")

        while connection_active:
            vad_state = session.vad.stream()
            try:
                packet = await vad_state.__anext__()
                if packet.is_speech and agent.call_context.get('tts_active', False):
                    logger.info("User interruption detected")
            except StopAsyncIteration:
                pass

            await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
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
                logger.info(f"LiveKit room disconnected gracefully. Reason: {ctx.room.disconnect_reason or 'unknown'}")
        except Exception as cleanup_e:
            logger.error(f"Error during cleanup: {cleanup_e}")




if __name__ == "__main__":
    logger.info(f"Starting {AGENT_NAME} for SIP trunk {SIP_INBOUND_TRUNK_ID}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME
        )
    )

