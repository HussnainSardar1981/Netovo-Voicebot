#!/usr/bin/env python3
"""
DTMF Detector for AudioSocket
Detects phone keypad tones (1-9, 0, *, #) from audio frames
Uses Goertzel algorithm for efficient frequency detection
"""

import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class DTMFDetector:
    """
    DTMF tone detector for AudioSocket voicebot
    Detects dual-tone multi-frequency signals from 8kHz audio
    """

    # DTMF frequency pairs (Hz)
    DTMF_FREQS = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
    }

    # Low and high frequency groups
    LOW_FREQS = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477]

    def __init__(self, sample_rate=8000, frame_duration_ms=20):
        """
        Initialize DTMF detector

        Args:
            sample_rate: Audio sample rate (8000 Hz for AudioSocket)
            frame_duration_ms: Frame duration in milliseconds (20ms)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = (sample_rate * frame_duration_ms) // 1000  # samples per frame

        # Detection thresholds
        self.energy_threshold = 100.0  # Minimum energy to consider signal
        self.tone_threshold = 0.3      # Relative magnitude threshold for tone detection
        self.min_tone_duration_ms = 40  # Minimum tone duration to register (2 frames)
        self.max_tone_duration_ms = 3000  # Maximum tone duration

        # State tracking
        self.current_digit = None
        self.digit_start_time = 0
        self.digit_frame_count = 0
        self.detected_digits = deque(maxlen=10)  # Last 10 detected digits
        self.last_detection_time = 0

        # Goertzel coefficient cache
        self.goertzel_coeffs = {}
        self._precompute_goertzel_coefficients()

        logger.info(f"DTMF detector initialized: {sample_rate}Hz, {frame_duration_ms}ms frames")

    def _precompute_goertzel_coefficients(self):
        """Precompute Goertzel coefficients for all DTMF frequencies"""
        all_freqs = set(self.LOW_FREQS + self.HIGH_FREQS)
        for freq in all_freqs:
            k = int(0.5 + (self.frame_size * freq) / self.sample_rate)
            w = (2.0 * np.pi * k) / self.frame_size
            self.goertzel_coeffs[freq] = 2.0 * np.cos(w)

    def _goertzel(self, samples, freq):
        """
        Goertzel algorithm for single frequency detection
        More efficient than FFT for detecting specific frequencies

        Args:
            samples: Audio samples (numpy array)
            freq: Target frequency

        Returns:
            Magnitude of the frequency component
        """
        coeff = self.goertzel_coeffs[freq]
        s_prev = 0.0
        s_prev2 = 0.0

        for sample in samples:
            s = sample + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s

        # Calculate magnitude
        power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
        return np.sqrt(power)

    def _detect_tone_pair(self, samples):
        """
        Detect which DTMF tone pair is present in the audio samples

        Args:
            samples: Audio samples (numpy array, float)

        Returns:
            Detected digit ('0'-'9', '*', '#') or None
        """
        # Calculate energy
        energy = np.sum(samples ** 2)
        if energy < self.energy_threshold:
            return None

        # Detect low frequency group
        low_magnitudes = {freq: self._goertzel(samples, freq) for freq in self.LOW_FREQS}
        max_low_freq = max(low_magnitudes, key=low_magnitudes.get)
        max_low_mag = low_magnitudes[max_low_freq]

        # Detect high frequency group
        high_magnitudes = {freq: self._goertzel(samples, freq) for freq in self.HIGH_FREQS}
        max_high_freq = max(high_magnitudes, key=high_magnitudes.get)
        max_high_mag = high_magnitudes[max_high_freq]

        # Check if both frequencies are strong enough
        avg_magnitude = (max_low_mag + max_high_mag) / 2.0
        if max_low_mag < self.tone_threshold * avg_magnitude or \
           max_high_mag < self.tone_threshold * avg_magnitude:
            return None

        # Find matching digit
        for digit, (low_freq, high_freq) in self.DTMF_FREQS.items():
            if low_freq == max_low_freq and high_freq == max_high_freq:
                return digit

        return None

    def process_frame(self, audio_bytes):
        """
        Process audio frame and detect DTMF tones

        Args:
            audio_bytes: Raw PCM audio bytes (int16 LE, 8kHz)

        Returns:
            Detected digit or None
        """
        # Convert bytes to numpy array (int16 → float)
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Detect tone in this frame
        detected_digit = self._detect_tone_pair(samples)

        # State machine for digit detection
        if detected_digit:
            if detected_digit == self.current_digit:
                # Same digit continues
                self.digit_frame_count += 1
            else:
                # New digit detected
                self.current_digit = detected_digit
                self.digit_frame_count = 1
                self.digit_start_time = 0  # Reset for new digit

            # Check if digit has been held long enough
            duration_ms = self.digit_frame_count * self.frame_duration_ms
            if duration_ms >= self.min_tone_duration_ms:
                # Valid digit detected
                if self.current_digit not in self.detected_digits or \
                   duration_ms > self.max_tone_duration_ms:
                    # First detection or held too long (new press)
                    logger.info(f"📱 DTMF detected: '{self.current_digit}' (duration: {duration_ms}ms)")
                    self.detected_digits.append(self.current_digit)
                    return self.current_digit

        else:
            # No tone detected - reset if we were tracking a digit
            if self.current_digit:
                duration_ms = self.digit_frame_count * self.frame_duration_ms
                logger.debug(f"DTMF ended: '{self.current_digit}' after {duration_ms}ms")
                self.current_digit = None
                self.digit_frame_count = 0

        return None

    def reset(self):
        """Reset detector state"""
        self.current_digit = None
        self.digit_frame_count = 0
        self.digit_start_time = 0
        self.detected_digits.clear()
        logger.debug("DTMF detector reset")

    def get_detected_digits(self):
        """Get list of detected digits in order"""
        return list(self.detected_digits)


def test_dtmf_detector():
    """Test DTMF detector with synthetic tones"""
    import matplotlib.pyplot as plt

    logger.info("Testing DTMF detector with synthetic tones")

    detector = DTMFDetector(sample_rate=8000, frame_duration_ms=20)

    # Generate test tone for digit '5' (770Hz + 1336Hz)
    duration = 0.1  # 100ms
    sample_rate = 8000
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = np.sin(2 * np.pi * 770 * t) + np.sin(2 * np.pi * 1336 * t)
    tone = (tone * 10000).astype(np.int16)  # Scale to int16

    # Process frames
    frame_size = 160  # 20ms @ 8kHz
    for i in range(0, len(tone) - frame_size, frame_size):
        frame = tone[i:i + frame_size].tobytes()
        digit = detector.process_frame(frame)
        if digit:
            logger.info(f"Detected: {digit}")

    logger.info(f"All detected digits: {detector.get_detected_digits()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dtmf_detector()
