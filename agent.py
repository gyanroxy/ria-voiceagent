import os
import asyncio
import certifi

# Fix for macOS SSL Certificate errors
os.environ['SSL_CERT_FILE'] = certifi.where()

import logging
import json
from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    sarvam,
    deepgram,
    silero,
    noise_cancellation,
)
from livekit.agents import llm
from typing import Annotated, Optional

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-agent")

import config


class TransferFunctions(llm.ToolContext):
    def __init__(self, ctx: agents.JobContext, phone_number: str = None):
        super().__init__(tools=[])
        self.ctx = ctx
        self.phone_number = phone_number

    @llm.function_tool(description="Look up customer vehicle and pending credit bill details by phone number.")
    def lookup_user(self, phone: str):
        """
        Look up customer record at Roxy Automobiles.
        """
        logger.info(f"Looking up customer: {phone}")
        return f"Customer record found for {phone}. Name: Verified Customer. Dealership: Roxy Automobiles. Overdue Credit Limit: Active balance pending."

    @llm.function_tool(description="Transfer the call to Roxy Automobiles senior accounts team or manager.")
    async def transfer_call(self, destination: Optional[str] = None):
        """
        Transfer the call to senior accounts manager.
        """
        if destination is None:
            destination = config.DEFAULT_TRANSFER_NUMBER
            if not destination:
                return "Error: No default transfer number configured."

        if "@" not in destination:
            if config.SIP_DOMAIN:
                clean_dest = destination.replace("tel:", "").replace("sip:", "")
                destination = f"sip:{clean_dest}@{config.SIP_DOMAIN}"
            else:
                if not destination.startswith("tel:") and not destination.startswith("sip:"):
                    destination = f"tel:{destination}"
        elif not destination.startswith("sip:"):
            destination = f"sip:{destination}"

        logger.info(f"Transferring call to {destination}")

        participant_identity = None
        if self.phone_number:
            participant_identity = f"sip_{self.phone_number}"
        else:
            for p in self.ctx.room.remote_participants.values():
                participant_identity = p.identity
                break

        if not participant_identity:
            logger.error("Could not determine participant identity for transfer")
            return "Failed to transfer: could not identify caller."

        try:
            logger.info(f"Transferring participant {participant_identity} to {destination}")
            await self.ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self.ctx.room.name,
                    participant_identity=participant_identity,
                    transfer_to=destination,
                    play_dialtone=False
                )
            )
            return "Call transfer initiated successfully."
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return f"Error executing transfer: {e}"


class OutboundAssistant(Agent):
    """
    AI receptionist agent powered by Groq and Sarvam AI.
    """
    def __init__(self, tools: list, instructions: str = None) -> None:
        super().__init__(
            instructions=instructions or config.SYSTEM_PROMPT,
            tools=tools,
        )


async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the outbound voice agent.
    - Connects to LiveKit Room.
    - Dials out via Vobiz SIP Trunk.
    - Engages caller with Groq LLM + Sarvam STT/TTS.
    """
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to LiveKit room successfully.")

    # Parse phone number and metadata
    phone_number = None
    user_prompt = None
    voice_name = config.SARVAM_TTS_VOICE

    VALID_VOICES = {
        "kavya", "priya", "ritu", "neha", "pooja", "rohan", "simran", "roopa",
        "kavitha", "shruti", "suhani", "aditya", "ashutosh", "rahul", "amit",
        "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit", "kabir",
        "aayan", "shubh", "advait", "anand", "tanya", "tarun", "sunny", "mani",
        "gokul", "vijay", "mohit", "rehan", "soham", "rupali", "niharika"
    }

    if ctx.job.metadata:
        try:
            data = json.loads(ctx.job.metadata)
            phone_number = data.get("phone_number")
            user_prompt = data.get("user_prompt")
            if data.get("voice_id") and data.get("voice_id").lower() in VALID_VOICES:
                voice_name = data.get("voice_id").lower()
        except Exception:
            pass

    if ctx.room.metadata:
        try:
            data = json.loads(ctx.room.metadata)
            if data.get("phone_number"):
                phone_number = data.get("phone_number")
            if data.get("user_prompt"):
                user_prompt = data.get("user_prompt")
            if data.get("voice_id") and data.get("voice_id").lower() in VALID_VOICES:
                voice_name = data.get("voice_id").lower()
        except Exception:
            pass

    fnc_ctx = TransferFunctions(ctx, phone_number)

    # 1. Brain: LLM (Sarvam LLM / Groq LLM)
    llm_provider = os.getenv("LLM_PROVIDER", config.LLM_PROVIDER).lower()
    if llm_provider == "sarvam" or not os.getenv("GROQ_API_KEY"):
        logger.info(f"Initializing Sarvam Indian Multilingual LLM ({config.SARVAM_LLM_MODEL})...")
        agent_llm = sarvam.LLM(
            model=config.SARVAM_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
        )
    else:
        logger.info(f"Initializing Groq LLM ({config.GROQ_MODEL})...")
        agent_llm = openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            _strict_tool_schema=False,
        )

    # 2. STT: Deepgram or Sarvam AI
    stt_provider = os.getenv("STT_PROVIDER", getattr(config, "STT_PROVIDER", "deepgram")).lower()
    if (stt_provider == "deepgram" and os.getenv("DEEPGRAM_API_KEY")) or (not os.getenv("SARVAM_API_KEY") and os.getenv("DEEPGRAM_API_KEY")):
        logger.info(f"Initializing Deepgram STT (model: {config.DEEPGRAM_STT_MODEL}, language: {config.DEEPGRAM_STT_LANGUAGE})...")
        agent_stt = deepgram.STT(
            model=config.DEEPGRAM_STT_MODEL,
            language=config.DEEPGRAM_STT_LANGUAGE,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        )
    else:
        logger.info(f"Initializing Sarvam STT ({config.SARVAM_STT_MODEL})...")
        agent_stt = sarvam.STT(
            model=config.SARVAM_STT_MODEL,
            language=config.SARVAM_STT_LANGUAGE,
            api_key=os.getenv("SARVAM_API_KEY"),
        )

    # 3. TTS: Deepgram or Sarvam AI
    tts_provider = os.getenv("TTS_PROVIDER", getattr(config, "TTS_PROVIDER", "deepgram")).lower()
    if (tts_provider == "deepgram" and os.getenv("DEEPGRAM_API_KEY")) or (not os.getenv("SARVAM_API_KEY") and os.getenv("DEEPGRAM_API_KEY")):
        logger.info(f"Initializing Deepgram TTS (model: {config.DEEPGRAM_TTS_MODEL})...")
        agent_tts = deepgram.TTS(
            model=config.DEEPGRAM_TTS_MODEL,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        )
    else:
        logger.info(f"Initializing Sarvam TTS ({config.SARVAM_TTS_MODEL}, voice: {voice_name})...")
        agent_tts = sarvam.TTS(
            model=config.SARVAM_TTS_MODEL,
            speaker=voice_name,
            target_language_code=config.SARVAM_TTS_LANGUAGE,
            api_key=os.getenv("SARVAM_API_KEY"),
        )

    # Build session with ultra-fast sub-second turn handling
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.25,
            prefix_padding_duration=0.2,
        ),
        stt=agent_stt,
        llm=agent_llm,
        tts=agent_tts,
        turn_handling={
            "endpointing": {
                "mode": "fixed",
                "min_delay": 0.15,
                "max_delay": 0.35,
            },
            "preemptive_generation": {
                "enabled": True,
                "preemptive_tts": True,
            },
            "interruption": {
                "enabled": True,
                "min_duration": 0.25,
            },
        },
    )

    # Attach real-time diagnostic event listeners
    @session.on("user_input_transcribed")
    def on_user_transcript(event):
        try:
            if hasattr(event, "transcript") and event.transcript and event.transcript.strip():
                logger.info(f"🎤 [Caller Transcribed]: {event.transcript.strip()} (is_final={getattr(event, 'is_final', False)})")
        except Exception:
            pass

    @session.on("agent_state_changed")
    def on_agent_state(event):
        try:
            state_val = getattr(event, "new_state", getattr(event, "state", "unknown"))
            logger.info(f"🤖 [Agent State]: {state_val}")
        except Exception:
            pass

    @session.on("error")
    def on_agent_error(event):
        try:
            logger.error(f"❌ [Session Error]: {getattr(event, 'error', event)}")
        except Exception:
            pass

    @session.on("close")
    def on_session_close(event):
        try:
            logger.info(f"📴 [Call Ended / Room Closed]: {getattr(event, 'reason', event)}")
        except Exception:
            pass

    prompt_instructions = config.SYSTEM_PROMPT
    if user_prompt:
        prompt_instructions += f"\n\nAdditional Context for this call: {user_prompt}"

    await session.start(
        room=ctx.room,
        agent=OutboundAssistant(
            tools=list(fnc_ctx.function_tools.values()),
            instructions=prompt_instructions
        ),
        room_input_options=RoomInputOptions(
            close_on_disconnect=True,
        ),
    )

    # Determine dial-out vs waiting for existing SIP participant
    should_dial = False
    if phone_number:
        user_already_here = any(
            f"sip_{phone_number}" in p.identity or "sip_" in p.identity
            for p in ctx.room.remote_participants.values()
        )
        if not user_already_here:
            should_dial = True

    if should_dial:
        logger.info(f"Initiating outbound SIP call to {phone_number} via trunk {config.SIP_TRUNK_ID}...")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=config.SIP_TRUNK_ID,
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number}",
                    wait_until_answered=True,
                )
            )
            logger.info("Call answered by user! Waiting 0.6s for audio stabilization...")
            await asyncio.sleep(0.6)

            logger.info(f"Speaking initial greeting to caller: {config.INITIAL_GREETING}")
            session.say(config.INITIAL_GREETING, allow_interruptions=True)
            logger.info("Greeting spoken. Ria is actively listening.")

        except Exception as e:
            logger.error(f"Failed to place outbound call: {e}")
            ctx.shutdown()
    else:
        logger.info("Awaiting customer participant to pick up and connect audio...")
        try:
            participant = await ctx.wait_for_participant()
            logger.info(f"Customer answered and joined: {participant.identity}")
        except Exception as e:
            logger.warning(f"Wait for participant warning: {e}")

        # Telephony audio line stabilization pause
        await asyncio.sleep(0.6)
        logger.info(f"Speaking initial greeting to caller: {config.INITIAL_GREETING}")
        session.say(config.INITIAL_GREETING, allow_interruptions=True)
        logger.info("Greeting spoken. Ria is actively listening for customer response.")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )

