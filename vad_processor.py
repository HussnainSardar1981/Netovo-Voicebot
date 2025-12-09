"""
Voice Activity Detection (VAD) Processor using webrtcvad

Handles speech/silence detection with hysteresis to prevent flickering.
Designed for 16kHz audio with 20ms frames.
"""

import webrtcvad
import logging
from collections import deque
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VADProcessor:
    """
    Voice Activity Detection with state tracking and hysteresis.

    Key features:
    - Aggressiveness levels 0-3 (3 = most aggressive filtering)
    - Hysteresis: require N consecutive speech frames to start, M to end
    - State tracking: IDLE, SPEECH_DETECTED, SPEECH_ENDED
    - Frame buffering for smooth transitions
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        aggressiveness: int = 2,
        speech_start_frames: int = 3,
        speech_end_frames: int = 20
    ):
        """
        Initialize VAD processor.

        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000 Hz)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            speech_start_frames: Consecutive speech frames to trigger speech start
            speech_end_frames: Consecutive silence frames to trigger speech end
        """
        # Validate parameters
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000 Hz, got {sample_rate}")

        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms, got {frame_duration_ms}")

        if aggressiveness not in [0, 1, 2, 3]:
            raise ValueError(f"Aggressiveness must be 0-3, got {aggressiveness}")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames

        # Calculate frame size in bytes (16-bit PCM = 2 bytes per sample)
        samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        self.frame_size = samples_per_frame * 2  # 2 bytes per sample

        # Initialize VAD
        self.vad = webrtcvad.Vad(aggressiveness)

        # State tracking
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0

        # Frame buffer for smoothing
        self.frame_buffer = deque(maxlen=100)  # Keep last 2 seconds at 20ms frames

        logger.info(
            f"VAD initialized: {sample_rate}Hz, {frame_duration_ms}ms frames, "
            f"aggressiveness={aggressiveness}, frame_size={self.frame_size} bytes"
        )

    def process_frame(self, audio_frame: bytes) -> tuple[bool, bool]:
        """
        Process a single audio frame through VAD.

        Args:
            audio_frame: Raw PCM audio data (must be exactly frame_size bytes)

        Returns:
            Tuple of (is_speech, state_changed)
            - is_speech: True if current frame contains speech
            - state_changed: True if speech state changed (started or ended)

        Raises:
            ValueError: If frame size is incorrect
        """
        # Validate frame size
        if len(audio_frame) != self.frame_size:
            raise ValueError(
                f"Frame size mismatch: expected {self.frame_size} bytes, "
                f"got {len(audio_frame)} bytes"
            )

        # Run VAD on frame
        try:
            is_speech = self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False, False

        # Add to buffer
        self.frame_buffer.append(is_speech)

        # Apply hysteresis
        state_changed = False

        if is_speech:
            self.speech_frame_count += 1
            self.silence_frame_count = 0

            # Trigger speech start if threshold reached
            if not self.is_speaking and self.speech_frame_count >= self.speech_start_frames:
                self.is_speaking = True
                state_changed = True
                logger.debug(f"Speech STARTED (after {self.speech_frame_count} frames)")
        else:
            self.silence_frame_count += 1
            self.speech_frame_count = 0

            # Trigger speech end if threshold reached
            if self.is_speaking and self.silence_frame_count >= self.speech_end_frames:
                self.is_speaking = False
                state_changed = True
                logger.debug(f"Speech ENDED (after {self.silence_frame_count} silence frames)")

        return is_speech, state_changed

    def reset(self):
        """Reset VAD state (call between utterances)."""
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.frame_buffer.clear()
        logger.debug("VAD state reset")

    @property
    def current_state(self) -> str:
        """Get current VAD state as string."""
        if self.is_speaking:
            return "SPEAKING"
        elif self.speech_frame_count > 0:
            return "SPEECH_STARTING"
        elif self.silence_frame_count > 0:
            return "SPEECH_ENDING"
        else:
            return "IDLE"

    def get_speech_ratio(self, window_frames: int = 50) -> float:
        """
        Calculate ratio of speech frames in recent window.

        Args:
            window_frames: Number of recent frames to analyze

        Returns:
            Ratio of speech frames (0.0 to 1.0)
        """
        if not self.frame_buffer:
            return 0.0

        recent_frames = list(self.frame_buffer)[-window_frames:]
        speech_count = sum(1 for f in recent_frames if f)
        return speech_count / len(recent_frames)


class InterruptionDetector:
    """
    Detects user interruptions during AI speech.

    This is the CRITICAL component for turn-taking - must detect interruptions
    in < 400ms to feel natural.
    """

    def __init__(
        self,
        vad_processor: VADProcessor,
        interruption_threshold_ms: int = 400,
        on_interruption: Optional[Callable] = None
    ):
        """
        Initialize interruption detector.

        Args:
            vad_processor: VAD processor instance
            interruption_threshold_ms: Time threshold to trigger interruption (ms)
            on_interruption: Callback function when interruption detected
        """
        self.vad = vad_processor
        self.interruption_threshold_ms = interruption_threshold_ms
        self.on_interruption = on_interruption

        # Calculate threshold in frames
        self.interruption_threshold_frames = int(
            interruption_threshold_ms / vad_processor.frame_duration_ms
        )

        # State
        self.enabled = False
        self.interruption_detected = False

        logger.info(
            f"Interruption detector initialized: {interruption_threshold_ms}ms "
            f"({self.interruption_threshold_frames} frames)"
        )

    def enable(self):
        """Enable interruption detection (call when AI starts speaking)."""
        self.enabled = True
        self.interruption_detected = False
        self.vad.reset()
        logger.debug("Interruption detection ENABLED")

    def disable(self):
        """Disable interruption detection (call when AI stops speaking)."""
        self.enabled = False
        self.interruption_detected = False
        logger.debug("Interruption detection DISABLED")

    def check_frame(self, audio_frame: bytes) -> bool:
        """
        Check if frame contains interruption.

        Args:
            audio_frame: Audio frame to check

        Returns:
            True if interruption detected
        """
        if not self.enabled or self.interruption_detected:
            return False

        # Process through VAD
        is_speech, state_changed = self.vad.process_frame(audio_frame)

        # Check if speech started (user is interrupting)
        if state_changed and self.vad.is_speaking:
            # Interruption detected!
            self.interruption_detected = True
            logger.warning(
                f"INTERRUPTION DETECTED after {self.vad.speech_frame_count} frames "
                f"({self.vad.speech_frame_count * self.vad.frame_duration_ms}ms)"
            )

            # Call callback if provided
            if self.on_interruption:
                try:
                    self.on_interruption()
                except Exception as e:
                    logger.error(f"Error in interruption callback: {e}")

            return True

        return False
