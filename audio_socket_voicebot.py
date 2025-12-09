"""
AudioSocket Voicebot - Complete Integration
Combines AudioSocket server with VAD, ASR, Kokoro TTS, and Ollama LLM.
"""

import asyncio
import logging
import numpy as np
import time
import requests
from pathlib import Path
from typing import Optional

# Local imports
from audio_socket_server import AudioSocketServer, AudioSocketConnection
from vad_processor import VADProcessor
from vosk_asr import VoskASR
from audio_utils import resample_8khz_to_16khz
from kokoro_tts_audiosocket import KokoroTTSClient
from ollama_audiosocket import OllamaClient
from model_warmup import SharedModels, load_models
from dtmf_detector import DTMFDetector
from config_audiosocket import (
    AudioSocketConfig, AudioConfig, ConversationState,
    TurnTakingConfig, LLMConfig, ZabbixConfig
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSocketVoicebot:
    """
    Complete voicebot implementation using AudioSocket.

    Integrates:
    - AudioSocket bidirectional audio
    - VAD for speech detection
    - Vosk ASR for transcription
    - Ollama LLM for responses (with optional RAG)
    - Kokoro TTS for synthesis
    """

    def __init__(self, connection: AudioSocketConnection, customer_id: Optional[str] = None):
        """
        Initialize AudioSocket Voicebot

        Args:
            connection: AudioSocket connection
            customer_id: Customer ID for RAG retrieval (optional)
                        Example: "stuart_dean", "skisafe", "millennium_escalators"
                        If None, RAG is disabled (LLM-only mode)
        """
        self.connection = connection
        self.state = ConversationState.IDLE
        self.customer_id = customer_id

        # Connection ID for debugging concurrent calls
        self.connection_id = connection.peer_address

        # Initialize components (ALL values from config - no hardcoding!)
        self.vad = VADProcessor(
            sample_rate=AudioConfig.VAD_SAMPLE_RATE,
            frame_duration_ms=AudioConfig.VAD_FRAME_DURATION_MS,
            aggressiveness=AudioConfig.VAD_AGGRESSIVENESS,
            speech_start_frames=AudioConfig.VAD_SPEECH_START_FRAMES,
            speech_end_frames=AudioConfig.VAD_SPEECH_END_FRAMES
        )

        # Use shared pre-loaded Vosk model
        if not SharedModels.models_loaded:
            logger.error("Models not loaded! Call load_models() before creating AudioSocketVoicebot")
            raise RuntimeError("Shared models not loaded")

        self.asr = VoskASR(
            model=SharedModels.vosk_model,
            sample_rate=AudioConfig.VOSK_SAMPLE_RATE
        )

        # Use new modular TTS and LLM clients
        self.tts = KokoroTTSClient(
            shared_pipeline=SharedModels.kokoro_pipeline,
            shared_device=SharedModels.kokoro_device
        )

        # Initialize LLM client with optional customer_id for RAG
        # RAG will be automatically enabled if customer_id is provided
        self.llm = OllamaClient(customer_id=customer_id)

        # Conversation history for ticket creation
        self.conversation_messages = []

        # Audio buffers
        self.user_speech_buffer = bytearray()  # 8kHz buffer for user speech
        self.silence_frames = 0
        self.speech_frames = 0

        # Interruption detection
        self.interruption_enabled = TurnTakingConfig.INTERRUPTION_ENABLED
        self.consecutive_speech_frames = 0
        self.interruption_requested = False  # Flag to stop audio playback

        logger.info(f"[{self.connection_id}] AudioSocket Voicebot initialized (customer: {customer_id or 'None'})")

    async def start(self):
        """Start voicebot conversation"""
        try:
            # Set audio callback
            self.connection.on_audio_received = self._on_audio_frame

            # Greeting
            await self._speak("Hello! How can I help you today?", voice_type="greeting")

            # Keep running until connection closes
            while self.connection.active:
                await asyncio.sleep(0.1)

            logger.info("Voicebot conversation ended")

        except Exception as e:
            logger.error(f"Voicebot error: {e}", exc_info=True)

    def _on_audio_frame(self, pcm_data: bytes):
        """
        Handle incoming audio frame from user.

        Args:
            pcm_data: 320 bytes of int16 LE PCM @ 8kHz
        """
        try:
            # Check for interruption during AI speaking (CRITICAL: ECHO CANCELLATION)
            if self.state == ConversationState.AI_SPEAKING:
                if self.interruption_enabled:
                    energy = self._calculate_energy(pcm_data)

                    # DYNAMIC THRESHOLD: Multiply by 2.5x during AI speaking to ignore echo
                    # Echo from bot's own voice will have lower energy than real user speech
                    # 300 * 2.5 = 750 threshold during AI speaking (vs 300 during user turn)
                    dynamic_threshold = TurnTakingConfig.INTERRUPTION_ENERGY_THRESHOLD * 2.5

                    if energy > dynamic_threshold:
                        # High energy detected - confirm it's real speech with VAD
                        is_real_speech = self.vad.process_frame(pcm_data)

                        if is_real_speech:
                            self.consecutive_speech_frames += 1
                            if self.consecutive_speech_frames >= TurnTakingConfig.INTERRUPTION_CONSECUTIVE_FRAMES:
                                logger.info(f"🎤 User interrupted bot (energy: {energy:.0f} > {dynamic_threshold:.0f})")
                                self._handle_interruption()
                        else:
                            # High energy but not speech (noise)
                            self.consecutive_speech_frames = 0
                    else:
                        # Low energy - likely echo or silence
                        self.consecutive_speech_frames = 0
                return  # Don't process audio during bot speaking

            # Pre-filter: reject low-energy frames (channel noise) before VAD
            energy = self._calculate_energy(pcm_data)
            if energy < TurnTakingConfig.VAD_ENERGY_THRESHOLD:
                is_speech = False
                # Log first few rejections for debugging
                if not hasattr(self, '_noise_logged'):
                    self._noise_logged = 0
                if self._noise_logged < 5:
                    logger.debug(f"Frame rejected: energy {energy:.1f} < threshold {TurnTakingConfig.VAD_ENERGY_THRESHOLD}")
                    self._noise_logged += 1
            else:
                # Energy above threshold - run VAD
                is_speech = self.vad.process_frame(pcm_data)

            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0

                # Transition to USER_SPEAKING
                if self.state == ConversationState.IDLE:
                    logger.info("User started speaking")
                    self.state = ConversationState.USER_SPEAKING
                    self.user_speech_buffer.clear()

                # Accumulate speech
                if self.state == ConversationState.USER_SPEAKING:
                    self.user_speech_buffer.extend(pcm_data)

                    # Debug: Log buffer size every 50 frames
                    if self.speech_frames % 50 == 0:
                        logger.debug(f"Speech: {self.speech_frames} frames, buffer: {len(self.user_speech_buffer)} bytes")

                    # Max speech duration timeout (from config)
                    if self.speech_frames >= AudioConfig.MAX_SPEECH_FRAMES:
                        logger.warning(f"Max speech duration reached ({self.speech_frames} frames = {AudioConfig.MAX_SPEECH_FRAMES * 20 / 1000:.1f}s), forcing transcription")
                        asyncio.create_task(self._process_user_speech())

            else:
                self.silence_frames += 1

                # Debug: Log silence detection
                if self.state == ConversationState.USER_SPEAKING:
                    if self.silence_frames % 10 == 0:
                        logger.debug(f"Silence: {self.silence_frames} frames, buffer: {len(self.user_speech_buffer)} bytes")

                # End of speech detection (silence threshold from config)
                if self.state == ConversationState.USER_SPEAKING and self.silence_frames >= AudioConfig.SILENCE_FRAMES_TO_END_SPEECH:
                    logger.info(f"User finished speaking (silence detected after {self.silence_frames} frames = {self.silence_frames * 20}ms)")
                    asyncio.create_task(self._process_user_speech())

        except Exception as e:
            logger.error(f"Error processing audio frame: {e}", exc_info=True)

    def _calculate_energy(self, pcm_data: bytes) -> float:
        """Calculate audio energy for interruption detection"""
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    def _handle_interruption(self):
        """Handle user interruption during bot speech"""
        # Set flag to stop audio playback immediately
        self.interruption_requested = True
        self.state = ConversationState.IDLE
        self.consecutive_speech_frames = 0
        logger.info("❌ INTERRUPTION - Stopping bot speech")

    async def _process_user_speech(self):
        """Process accumulated user speech"""
        try:
            self.state = ConversationState.PROCESSING

            # Get speech buffer
            speech_8khz = bytes(self.user_speech_buffer)
            self.user_speech_buffer.clear()

            # Check minimum speech length (from config)
            if len(speech_8khz) < AudioConfig.MIN_SPEECH_BYTES:
                logger.warning(f"Speech too short ({len(speech_8khz)} bytes < {AudioConfig.MIN_SPEECH_BYTES}), ignoring")
                self.state = ConversationState.IDLE
                self.speech_frames = 0
                return

            # Resample 8kHz -> 16kHz for Vosk
            speech_16khz = resample_8khz_to_16khz(speech_8khz)

            # Transcribe with Vosk
            logger.info("Transcribing speech...")
            transcript = await asyncio.to_thread(
                self.asr.transcribe_audio,
                speech_16khz,
                emit_final=True
            )

            if not transcript or transcript.strip() == "":
                logger.warning("Empty transcription")
                self.state = ConversationState.IDLE
                return

            logger.info(f"User said: {transcript}")

            # Track user message
            self.conversation_messages.append({
                'role': 'user',
                'content': transcript
            })

            # Generate LLM response WITH TRUE STREAMING (speaks while generating)
            logger.info("Generating response (streaming)...")
            full_response = ""

            # Create generator in thread, then iterate and speak each sentence IMMEDIATELY
            def get_generator():
                """Get the streaming generator"""
                return self.llm.generate_response_streaming(transcript)

            # Get the generator
            sentence_generator = await asyncio.to_thread(get_generator)

            # NOW iterate through sentences and speak each one as it arrives
            # This is TRUE streaming - we speak sentence 1 while LLM generates sentence 2
            for sentence in sentence_generator:
                # Got a sentence! Speak it immediately
                await self._speak(sentence)
                full_response += " " + sentence

            # Track assistant response for ticket creation
            self.conversation_messages.append({
                'role': 'assistant',
                'content': full_response.strip()
            })

            # Check for ticket creation markers or time-based trigger
            await self._check_and_create_ticket(full_response.strip())

            # Return to IDLE
            self.state = ConversationState.IDLE

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            self.state = ConversationState.IDLE

    async def _check_and_create_ticket(self, response_text: str):
        """Check if ticket should be created based on response"""
        # Placeholder for ticket creation logic
        # This can be extended later to create support tickets
        pass

    async def _speak(self, text: str, voice_type: str = "default"):
        """
        Synthesize and send speech to user.

        Args:
            text: Text to speak
            voice_type: Voice type (empathetic, technical, greeting, default)
        """
        try:
            # Check if connection is still active before speaking
            if not self.connection.active:
                logger.warning("Connection closed - cannot speak")
                return

            self.state = ConversationState.AI_SPEAKING
            self.interruption_requested = False  # Reset interruption flag
            logger.info(f"Bot speaking: {text}")

            # Synthesize with Kokoro (using af_heart voice with appropriate voice type)
            audio_pcm_8khz = await asyncio.to_thread(
                self.tts.synthesize,
                text,
                voice_type=voice_type
            )

            if not audio_pcm_8khz:
                logger.error("TTS synthesis failed")
                self.state = ConversationState.IDLE
                return

            # Send audio in 320-byte frames (20ms @ 8kHz)
            frame_size = AudioSocketConfig.FRAME_SIZE
            frames_sent = 0
            total_frames = len(audio_pcm_8khz) // frame_size

            for i in range(0, len(audio_pcm_8khz), frame_size):
                # Check for interruption BEFORE sending each frame
                if self.interruption_requested:
                    logger.info(f"🛑 Audio playback stopped by user interruption (sent {frames_sent}/{total_frames} frames)")
                    self.state = ConversationState.IDLE
                    self.speech_frames = 0  # Reset for next user input
                    return

                frame = audio_pcm_8khz[i:i + frame_size]

                # Pad last frame if needed
                if len(frame) < frame_size:
                    frame += b'\x00' * (frame_size - len(frame))

                # Send frame
                await self.connection.send_audio(frame)
                frames_sent += 1

                # Throttle to real-time (20ms per frame)
                await asyncio.sleep(0.020)

            logger.info("Finished speaking")
            self.state = ConversationState.IDLE

        except Exception as e:
            logger.error(f"Error speaking: {e}", exc_info=True)
            self.state = ConversationState.IDLE

    async def start_alert_call(self, call_id: str):
        """
        Handle Zabbix alert call with DTMF response

        Args:
            call_id: Zabbix alert call ID
        """
        try:
            logger.info(f"🚨 Starting Zabbix alert call: {call_id}")

            # Query Zabbix server for alert data
            alert_data = await self._get_alert_data(call_id)
            if not alert_data:
                logger.error(f"Failed to get alert data for {call_id}")
                await self._speak("Alert data unavailable. Please contact support.")
                return

            # Log alert details
            logger.info(f"Alert ID: {alert_data['alert_id']}")
            logger.info(f"Hostname: {alert_data['hostname']}")
            logger.info(f"Trigger: {alert_data['trigger']}")
            logger.info(f"Severity: {alert_data['severity']}")

            # Create alert message
            alert_message = (
                f"Hello, this is Alexis from Netovo monitoring. "
                f"We have a {alert_data['severity']} severity alert on {alert_data['hostname']}: "
                f"{alert_data['trigger']}. "
                f"Press 1 to acknowledge this alert, or press 2 if you are not available."
            )

            # Speak alert message
            await self._speak(alert_message, voice_type="technical")

            # Wait for DTMF response
            logger.info("Waiting for DTMF response...")
            dtmf_digit = await self._wait_for_dtmf(timeout=ZabbixConfig.DTMF_WAIT_TIMEOUT)

            # Process DTMF response
            if dtmf_digit == '1':
                logger.info(f"✅ Alert {alert_data['alert_id']} acknowledged")
                response_msg = "Alert acknowledged. Thank you. A support ticket will be created for tracking."
                await self._send_dtmf_response(call_id, alert_data['alert_id'], '1')

            elif dtmf_digit == '2':
                logger.info(f"❌ Technician not available for {alert_data['alert_id']}")
                response_msg = "Understood. This alert will be escalated to the manager."
                await self._send_dtmf_response(call_id, alert_data['alert_id'], '2')

            else:
                logger.warning(f"⚠️ No valid DTMF response for {alert_data['alert_id']}")
                response_msg = "No response received. This alert will be escalated."
                # Don't send DTMF response if none received

            # Speak confirmation
            await self._speak(response_msg, voice_type="default")

            logger.info(f"🚨 Alert call completed: {call_id}")

        except Exception as e:
            logger.error(f"Error handling alert call: {e}", exc_info=True)

    async def _get_alert_data(self, call_id: str) -> Optional[dict]:
        """Query Zabbix alert server for alert data"""
        try:
            url = f"{ZabbixConfig.ALERT_SERVER_URL}/alert-data/{call_id}"
            response = await asyncio.to_thread(
                requests.get,
                url,
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('alert_data')
                else:
                    logger.error(f"Alert server error: {data.get('error')}")
                    return None
            else:
                logger.error(f"Failed to get alert data: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error querying alert data: {e}")
            return None

    async def _wait_for_dtmf(self, timeout: int = 30) -> Optional[str]:
        """
        Wait for DTMF tone from user

        Args:
            timeout: Timeout in seconds

        Returns:
            Detected DTMF digit or None
        """
        try:
            # Create DTMF detector
            dtmf_detector = DTMFDetector(
                sample_rate=ZabbixConfig.DTMF_SAMPLE_RATE,
                frame_duration_ms=ZabbixConfig.DTMF_FRAME_DURATION_MS
            )
            dtmf_detector.energy_threshold = ZabbixConfig.DTMF_ENERGY_THRESHOLD
            dtmf_detector.tone_threshold = ZabbixConfig.DTMF_TONE_THRESHOLD
            dtmf_detector.min_tone_duration_ms = ZabbixConfig.DTMF_MIN_DURATION_MS

            detected_digit = None
            start_time = time.time()

            # Temporary audio callback for DTMF detection
            def dtmf_callback(pcm_data: bytes):
                nonlocal detected_digit
                digit = dtmf_detector.process_frame(pcm_data)
                if digit:
                    detected_digit = digit

            # Set DTMF callback
            old_callback = self.connection.on_audio_received
            self.connection.on_audio_received = dtmf_callback

            # Wait for DTMF or timeout
            while detected_digit is None and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            # Restore original callback
            self.connection.on_audio_received = old_callback

            if detected_digit:
                logger.info(f"📱 DTMF detected: {detected_digit}")
            else:
                logger.warning(f"⏱️ DTMF timeout after {timeout}s")

            return detected_digit

        except Exception as e:
            logger.error(f"Error waiting for DTMF: {e}", exc_info=True)
            return None

    async def _send_dtmf_response(self, call_id: str, alert_id: str, dtmf_response: str):
        """Send DTMF response to Zabbix alert server"""
        try:
            url = f"{ZabbixConfig.ALERT_SERVER_URL}/dtmf-response"
            data = {
                'call_id': call_id,
                'dtmf_response': dtmf_response,
                'alert_id': alert_id
            }

            response = await asyncio.to_thread(
                requests.post,
                url,
                json=data,
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"✅ DTMF response sent to alert server: {dtmf_response}")
            else:
                logger.warning(f"⚠️ Alert server responded with HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Failed to send DTMF response: {e}")


async def main():
    """Main entry point"""
    logger.info("Starting AudioSocket Voicebot Server...")

    # Load models once at startup (model warmup)
    load_models()

    # Connection handler
    def on_connection(connection: AudioSocketConnection):
        """Handle new AudioSocket connection"""
        logger.info(f"New call from {connection.peer_address}")

        # Wait briefly for UUID message from Asterisk
        async def handle_voicebot():
            # Give time for UUID to be received
            await asyncio.sleep(0.2)

            # ===== CUSTOMER IDENTIFICATION FOR RAG =====
            # TODO: Implement customer identification logic based on:
            # - Caller phone number (from connection.peer_address)
            # - Asterisk variables (from connection.session_uuid)
            # - Database lookup
            # - DID (Direct Inward Dialing) number called
            #
            # For now, set customer_id to "stuart_dean" for testing
            customer_id = "stuart_dean"  # RAG enabled for testing

            # Other customer options:
            # customer_id = "skisafe"      # Test with skisafe collection
            # customer_id = None           # Disable RAG (LLM-only mode)

            # Example: Customer identification from phone number
            # caller_number = connection.peer_address[0] if connection.peer_address else None
            # customer_id = lookup_customer_by_phone(caller_number)

            logger.info(f"Customer ID for this call: {customer_id or 'None (RAG disabled)'}")

            # Create voicebot for this connection with customer_id
            voicebot = AudioSocketVoicebot(connection, customer_id=customer_id)

            # Check if this is a Zabbix alert call
            call_id = None
            if connection.session_uuid:
                try:
                    uuid_str = connection.session_uuid.decode('utf-8', errors='ignore')
                    if uuid_str.startswith(ZabbixConfig.ALERT_CALL_ID_PREFIX):
                        call_id = uuid_str
                        logger.info(f"ZABBIX ALERT CALL detected: {call_id}")
                except:
                    pass

            # Route to appropriate handler
            if call_id:
                # Zabbix alert call
                await voicebot.start_alert_call(call_id)
            else:
                # Normal customer call
                await voicebot.start()

        # Start voicebot handler in background
        asyncio.create_task(handle_voicebot())

    # Create and start server
    server = AudioSocketServer(
        host=AudioSocketConfig.HOST,
        port=AudioSocketConfig.PORT,
        on_connection=on_connection
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
