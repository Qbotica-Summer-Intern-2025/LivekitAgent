from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any
from datetime import datetime

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    silero,
    noise_cancellation,
)
#from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("logistics-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

class LogisticsAgent(Agent):
    def __init__(self, *, dial_info: dict[str, Any]):
        super().__init__(
            instructions=f"""
            You are a freight logistics agent calling companies to collect truckload shipping quotes.
            Ask about:
            - Origin and destination cities
            - Truck type (reefer, dry van, flatbed, etc.)
            - Quoted price
            - Pickup window or availability

            Record the quote using the record_quote tool.
            Be polite, clear, and end the call if the provider asks to be contacted via email or declines.
            """
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        self.phone_number = dial_info.get("phone_number", "unknown")
        self.quotes = []

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

    @function_tool()
    async def record_quote(self, ctx: RunContext, origin: str, destination: str, quote: str, truck_type: str):
        self.quotes.append({
            "origin": origin,
            "destination": destination,
            "quote": quote,
            "truck_type": truck_type,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Recorded quote: {origin} to {destination} @ {quote} ({truck_type})")
        return "Thanks, your quote has been recorded."

    @function_tool()
    async def end_call(self, ctx: RunContext):
        await self.save_results(ctx)
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()
        await self.hangup()

    async def save_results(self, ctx: RunContext):
        results = {
            "phone_number": self.phone_number,
            "quotes": self.quotes,
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs("quotes", exist_ok=True)
        filename = f"quotes/quote_{self.phone_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved quote data to {filename}")

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()

    try:
        dial_info = json.loads(ctx.job.metadata)
        phone_number = dial_info.get("phone_number")
        participant_identity = phone_number
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid metadata: {e}")
        ctx.shutdown()
        return

    agent = LogisticsAgent(dial_info=dial_info)

    session = AgentSession(
        turn_detection=None,
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        llm=openai.LLM(model="gpt-4o"),
    )

    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
    )

    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                wait_until_answered=True,
            )
        )

        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"Participant joined: {participant.identity}")
        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"Error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="logistics-agent",
        )
    )
