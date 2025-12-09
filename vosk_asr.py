"""
Streaming Vosk ASR (Automatic Speech Recognition)

Real-time speech recognition with partial and final results.
Optimized for low-latency turn-taking conversations.
"""

import json
import logging
from typing import Optional, Callable
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)


class VoskASR:
    """
    Streaming ASR using Vosk.

    Key features:
    - Streaming recognition (frame-by-frame, not batch)
    - Partial results for real-time feedback
    - Final results for complete utterances
    - Automatic punctuation and capitalization
    - Low memory footprint
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Model] = None,
        sample_rate: int = 16000,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize Vosk ASR.

        Args:
            model_path: Path to Vosk model directory (if model not provided)
            model: Pre-loaded Vosk model (preferred for avoiding blocking)
            sample_rate: Audio sample rate (must match audio input)
            on_partial: Callback for partial results (called frequently)
            on_final: Callback for final results (called on utterance end)
        """
        self.sample_rate = sample_rate
        self.on_partial = on_partial
        self.on_final = on_final

        # Use pre-loaded model if provided, otherwise load from path
        if model is not None:
            logger.info("Using pre-loaded Vosk model (shared)")
            self.model = model
            self.model_path = None
        elif model_path is not None:
            logger.info(f"Loading Vosk model from {model_path}...")
            try:
                self.model = Model(model_path)
                self.model_path = model_path
                logger.info("Vosk model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Vosk model: {e}")
                raise
        else:
            raise ValueError("Either model_path or model must be provided")

        # Create recognizer
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)  # Enable word-level timestamps

        # State
        self.is_processing = False
        self.current_utterance = ""
        self.final_results = []

        logger.info(f"Vosk ASR initialized: {sample_rate}Hz")

    def process_audio(self, audio_data: bytes) -> Optional[dict]:
        """
        Process audio chunk and return recognition result.

        Args:
            audio_data: Raw PCM audio data (16-bit, mono)

        Returns:
            Recognition result dict if available, None otherwise
            Format: {"partial": str} or {"text": str, "result": [...]}
        """
        if not self.is_processing:
            return None

        try:
            # Feed audio to recognizer
            if self.recognizer.AcceptWaveform(audio_data):
                # Final result (end of utterance detected)
                result = json.loads(self.recognizer.Result())

                if result.get("text"):
                    final_text = result["text"]
                    self.current_utterance = ""
                    self.final_results.append(final_text)

                    logger.info(f"Final ASR result: '{final_text}'")

                    # Call callback
                    if self.on_final:
                        try:
                            self.on_final(final_text)
                        except Exception as e:
                            logger.error(f"Error in final callback: {e}")

                    return {"text": final_text, "result": result.get("result", [])}
            else:
                # Partial result (still speaking)
                result = json.loads(self.recognizer.PartialResult())
                partial_text = result.get("partial", "")

                if partial_text and partial_text != self.current_utterance:
                    self.current_utterance = partial_text
                    logger.debug(f"Partial ASR: '{partial_text}'")

                    # Call callback
                    if self.on_partial:
                        try:
                            self.on_partial(partial_text)
                        except Exception as e:
                            logger.error(f"Error in partial callback: {e}")

                    return {"partial": partial_text}

        except Exception as e:
            logger.error(f"ASR processing error: {e}")

        return None

    def start_recognition(self):
        """Start ASR processing (enable frame processing)."""
        self.is_processing = True
        self.current_utterance = ""
        logger.debug("ASR recognition STARTED")

    def stop_recognition(self, emit_final: bool = True) -> Optional[str]:
        """
        Stop ASR processing and optionally get final result.

        Args:
            emit_final: If True, emit final result and call callbacks.
                       If False, suppress logging/callbacks (for cleanup)

        Returns:
            Final recognized text if available, None otherwise
        """
        self.is_processing = False

        try:
            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get("text", "")

            if final_text:
                self.final_results.append(final_text)

                # Only log and call callbacks if emit_final=True
                if emit_final:
                    logger.info(f"Final ASR on stop: '{final_text}'")

                    # Call callback
                    if self.on_final:
                        try:
                            self.on_final(final_text)
                        except Exception as e:
                            logger.error(f"Error in final callback: {e}")

                return final_text
        except Exception as e:
            logger.error(f"Error getting final result: {e}")

        logger.debug(f"ASR recognition STOPPED (emit_final={emit_final})")
        return None

    def reset(self):
        """Reset recognizer state (call between utterances)."""
        # Create new recognizer (Vosk doesn't have explicit reset)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        self.current_utterance = ""
        logger.debug("ASR recognizer reset")

    def transcribe_audio(self, audio_data: bytes, emit_final: bool = True) -> str:
        """
        Batch transcription of complete audio buffer.
        Used for offline/batch processing of accumulated speech.

        Args:
            audio_data: Complete audio buffer (16-bit PCM @ self.sample_rate)
            emit_final: Whether to emit final result and trigger callbacks

        Returns:
            Transcribed text string
        """
        try:
            # Create a fresh recognizer for this transcription
            recognizer = KaldiRecognizer(self.model, self.sample_rate)
            recognizer.SetWords(True)

            # Process entire audio buffer
            recognizer.AcceptWaveform(audio_data)

            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            transcript = final_result.get("text", "").strip()

            if transcript and emit_final:
                logger.info(f"Batch transcription: '{transcript}'")
                self.final_results.append(transcript)

                # Call callback if registered
                if self.on_final:
                    try:
                        self.on_final(transcript)
                    except Exception as e:
                        logger.error(f"Error in final callback: {e}")

            return transcript

        except Exception as e:
            logger.error(f"Batch transcription error: {e}", exc_info=True)
            return ""

    def get_recent_results(self, count: int = 5) -> list[str]:
        """
        Get recent final results.

        Args:
            count: Number of recent results to return

        Returns:
            List of recent final results
        """
        return self.final_results[-count:]


class ASRManager:
    """
    High-level ASR manager with state-aware processing.

    This integrates ASR with the turn-taking state machine to prevent
    echo (bot transcribing its own speech).
    """

    def __init__(self, vosk_asr: VoskASR):
        """
        Initialize ASR manager.

        Args:
            vosk_asr: Vosk ASR instance
        """
        self.asr = vosk_asr
        self.enabled = True  # Global enable/disable
        self.processing_allowed = True  # State-based gating

    def enable(self):
        """Enable ASR globally."""
        self.enabled = True
        logger.debug("ASR ENABLED globally")

    def disable(self, emit_final: bool = True):
        """
        Disable ASR globally.

        Args:
            emit_final: If True, emit final result on stop.
                       If False, suppress final result (for cleanup)
        """
        self.enabled = False
        self.asr.stop_recognition(emit_final=emit_final)
        logger.debug(f"ASR DISABLED globally (emit_final={emit_final})")

    def allow_processing(self):
        """
        Allow ASR processing (called when entering USER_SPEAKING state).

        This is the KEY echo prevention mechanism - only process ASR when
        we're in a state where the user should be speaking.
        """
        self.processing_allowed = True
        if self.enabled:
            self.asr.start_recognition()
            logger.debug("ASR processing ALLOWED (user turn)")

    def block_processing(self):
        """
        Block ASR processing (called when entering AI_SPEAKING state).

        This prevents the bot from transcribing its own speech!
        """
        self.processing_allowed = False
        self.asr.stop_recognition()
        logger.debug("ASR processing BLOCKED (AI turn - echo prevention)")

    def process_audio(self, audio_data: bytes) -> Optional[dict]:
        """
        Process audio through ASR with state checking.

        Args:
            audio_data: Raw PCM audio data

        Returns:
            Recognition result if processing allowed, None otherwise
        """
        if not self.enabled or not self.processing_allowed:
            return None

        return self.asr.process_audio(audio_data)

    def reset(self):
        """Reset ASR state."""
        self.asr.reset()
