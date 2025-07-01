import asyncio
import logging
import os
import json
import random
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SlackAgent")

AGENT_NAME = "slack-agent"
LIVEKIT_ROOM_PREFIX = "call-"

load_dotenv()
required_env = [
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
    "SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID"
]
for var in required_env:
    if not os.getenv(var):
        raise ValueError(f"{var} missing in .env")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

slack_client = WebClient(token=SLACK_BOT_TOKEN)

async def send_slack_message(message: str):
    """Send a message to the configured Slack channel."""
    try:
        response = slack_client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=message
        )
        if response["ok"]:
            logger.info(f"Message sent to channel {SLACK_CHANNEL_ID}: {message}")
        else:
            logger.error(f"Failed to send Slack message: {response['error']}")
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}")
        if e.response.get("error") == "ratelimited":
            retry_after = int(e.response.headers.get("retry-after", 1))
            logger.warning(f"Rate limited on chat.postMessage, waiting {retry_after} seconds")
            await asyncio.sleep(retry_after)

async def receive_slack_messages(ctx: JobContext, last_timestamp: str = None):
    """Poll the Slack channel for new messages and publish them to the LiveKit room."""
    try:
        response = slack_client.conversations_history(
            channel=SLACK_CHANNEL_ID,
            oldest=last_timestamp,
            limit=10
        )
        if response["ok"]:
            messages = response["messages"]
            new_timestamp = last_timestamp
            for msg in reversed(messages):
                if not msg.get("subtype") and "text" in msg:
                    message = msg["text"]
                    ts = msg["ts"]
                    if not last_timestamp or float(ts) > float(last_timestamp):
                        logger.info(f"Received Slack message: {message} (ts: {ts})")
                        try:
                            payload = json.dumps({"slack_message": message}).encode('utf-8')
                            await ctx.room.local_participant.publish_data(payload=payload)
                            logger.info(f"Published Slack message to LiveKit room: {message}")
                            new_timestamp = ts
                        except Exception as e:
                            logger.error(f"Error publishing Slack message to LiveKit room: {e}")
            return new_timestamp
        else:
            logger.error(f"Failed to fetch Slack messages: {response['error']}")
            return last_timestamp
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}")
        if e.response.get("error") == "ratelimited":
            retry_after = int(e.response.headers.get("retry-after", 10))
            logger.warning(f"Rate limited on conversations.history, waiting {retry_after} seconds")
            await asyncio.sleep(retry_after)
        return last_timestamp

def on_data_received(packet: rtc.DataPacket, ctx: JobContext):
    """Synchronous callback for data_received event."""
    logger.debug(f"Data packet received: {packet}")
    try:
        payload = packet.data.decode('utf-8')
        logger.debug(f"Decoded payload: {payload}")
        data = json.loads(payload)
        message = data.get("message")
        if message:
            logger.info(f"Received data message: message={message}")
            asyncio.create_task(send_slack_message(message))
        else:
            logger.warning(f"Invalid data message: {payload}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse data message: {payload}")
    except Exception as e:
        logger.error(f"Error processing data message: {e}")

async def entrypoint(ctx: JobContext):
    """Entrypoint for the Slack agent to join the LiveKit room and handle messages."""
    try:
        logger.info(f"Starting {AGENT_NAME} entrypoint for room: {ctx.room.name}")
        await ctx.connect(auto_subscribe=True)
        logger.info(f"Slack agent connected to {LIVEKIT_URL}, room: {ctx.room.name}")
        logger.debug(f"Room connection state: {ctx.room.connection_state}")

        ctx.room.on("data_received", lambda packet: on_data_received(packet, ctx))

        # Start polling for Slack messages
        last_timestamp = None
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            last_timestamp = await receive_slack_messages(ctx, last_timestamp)
            await asyncio.sleep(60)  # Poll every 60 seconds to avoid rate limits
            logger.debug("Slack agent still connected")

    except Exception as e:
        logger.error(f"Error in entrypoint: {e}")
        raise
    finally:
        try:
            if ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                await ctx.room.disconnect()
                logger.info("Slack agent disconnected from room")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def run_app_with_room(room_name: str = None):
    """Run the app with the specified room name."""
    default_room = os.getenv("LIVEKIT_ROOM_NAME", f"{LIVEKIT_ROOM_PREFIX}{random.randint(1000, 9999)}")
    room = room_name or default_room
    logger.info(f"Starting {AGENT_NAME} for room: {room}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
        )
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=f"Run {AGENT_NAME}")
    parser.add_argument("action", choices=["connect"], help="Action to perform")
    parser.add_argument("--room", help="LiveKit room name")
    args = parser.parse_args()

    if args.action == "connect":
        run_app_with_room(args.room)