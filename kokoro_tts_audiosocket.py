#!/usr/bin/env python3
"""
Kokoro TTS Client for AudioSocket
Professional Text-to-Speech using Kokoro - adapted from AGI version
"""

import os
import time
import uuid
import html
import logging
import subprocess
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from kokoro import KPipeline
from config_audiosocket import TEMP_DIR

logger = logging.getLogger(__name__)


class KokoroTTSClient:
    """Professional Kokoro TTS Client for AudioSocket"""

    def __init__(self, shared_pipeline=None, shared_device=None):
        """
        Initialize Kokoro TTS client.

        Args:
            shared_pipeline: Pre-loaded KPipeline instance (for model warmup)
            shared_device: Device string ('cuda' or 'cpu')
        """
        try:
            # Use shared pipeline if provided (model warmup), otherwise load new
            if shared_pipeline is not None:
                logger.info("Using shared pre-loaded Kokoro pipeline")
                self.pipeline = shared_pipeline
                self.device = shared_device
            else:
                logger.info("Loading new Kokoro TTS pipeline...")
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"GPU detected for TTS: {gpu_name}")
                    self.pipeline = KPipeline(lang_code='a', device=self.device)
                else:
                    logger.info("Using CPU for TTS")
                    self.pipeline = KPipeline(lang_code='a')

            # Voice configuration (from AGI)
            self.voice_name = "af_heart"  # Most human-like voice
            self.sample_rate = 24000      # Kokoro native
            self.target_sample_rate = 8000  # AudioSocket/Asterisk
            self.speed = 0.92             # Natural speaking pace

            # Voice mapping
            self.voice_mapping = {
                "af_sarah": "af_sarah",
                "af_bella": "af_bella",
                "af_jessica": "af_jessica",
                "af_nova": "af_nova",
                "af_sky": "af_sky",
                "af_heart": "af_heart",  # Default
                "af_alloy": "af_alloy"
            }

            self.kokoro_voice = self.voice_mapping.get(self.voice_name, "af_heart")

            logger.info(f"✅ Kokoro TTS ready: voice={self.kokoro_voice}, device={self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            raise

    def _get_voice_speed(self, voice_type):
        """Get speech speed based on voice type for natural conversation flow"""
        speed_map = {
            "empathetic": 0.88,    # Slower = more empathetic
            "technical": 0.94,     # Slightly slower for clarity
            "greeting": 0.90,      # Warm greeting pace
            "default": 0.92        # Natural pace
        }
        return speed_map.get(voice_type, 0.92)

    def _enhance_text_for_speech(self, text, voice_type="default"):
        """Enhance text for more natural speech with pronunciation fixes"""
        # Escape problematic characters
        safe_text = html.escape(text, quote=False)

        # Acronym spell-outs
        pronunciation_fixes = {
            "AGI": "A-G-I",
            "API": "A-P-I",
            "VoIP": "Voice over I-P",
            "SIP": "S-I-P",
        }
        for original, phonetic in pronunciation_fixes.items():
            safe_text = safe_text.replace(original, phonetic)

        # Company names
        for v in ("NETOVO", "Netovo", "netovo"):
            safe_text = safe_text.replace(v, "Netovo")

        # Basic text normalization
        safe_text = safe_text.replace("&", " and ")
        safe_text = safe_text.replace("%", " percent ")
        safe_text = safe_text.replace("@", " at ")
        safe_text = safe_text.replace("#", " number ")

        # Common tech pronunciations
        safe_text = safe_text.replace("24/7", "twenty-four seven")
        safe_text = safe_text.replace("3CX", "three C X")

        # Light pausing for empathetic tone
        if voice_type == "empathetic":
            for w in ("sorry", "understand", "apologize", "help"):
                safe_text = safe_text.replace(f" {w} ", f" {w}, ")

        return safe_text

    def synthesize(self, text: str, voice_type: str = "default", voice: str = None) -> bytes:
        """
        Synthesize speech from text using Kokoro TTS.
        Returns PCM audio bytes at 8kHz (for AudioSocket).

        Args:
            text: Text to synthesize
            voice_type: Voice type (empathetic, technical, greeting, default)
            voice: Voice name override (af_sky, af_bella, af_heart, etc.)

        Returns:
            PCM audio data at 8kHz (int16 LE)
        """
        temp_24k = None
        temp_8k = None

        try:
            # Use voice override or default
            kokoro_voice = self.voice_mapping.get(voice or self.kokoro_voice, "af_heart")

            # Enhance text for natural speech
            enhanced_text = self._enhance_text_for_speech(text, voice_type)

            logger.info(f" Kokoro TTS: voice={kokoro_voice}, type={voice_type}")
            logger.debug(f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # Generate audio using KPipeline (yields chunks at 24kHz)
            # CRITICAL: Use lock for thread safety across concurrent calls
            # This ensures multiple calls don't interfere with the shared neural network
            from model_warmup import SharedModels

            with SharedModels.kokoro_lock:
                generator = self.pipeline(enhanced_text, voice=kokoro_voice)

                # Collect audio chunks
                audio_chunks = []
                for i, (gs, ps, audio_chunk) in enumerate(generator):
                    audio_chunks.append(audio_chunk)

            if not audio_chunks:
                logger.error("No audio generated from Kokoro")
                return b''

            # Concatenate chunks (outside lock - this is just numpy operations)
            full_audio = np.concatenate(audio_chunks)

            # Create temp directory if needed
            TEMP_DIR.mkdir(parents=True, exist_ok=True)

            # Generate unique temp filenames
            unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            temp_24k = str(TEMP_DIR / f"kokoro_temp_{unique_id}.wav")
            temp_8k = str(TEMP_DIR / f"kokoro_tts_{unique_id}.wav")

            # Save at native sample rate (24kHz)
            sf.write(temp_24k, full_audio, self.sample_rate, subtype='PCM_16')

            logger.debug(f"Generated audio: {len(full_audio)} samples at {self.sample_rate}Hz")

            # Convert to 8kHz using sox (EXACT AGI approach)
            sox_cmd = [
                'sox', temp_24k,
                '-r', str(self.target_sample_rate),  # 8kHz for AudioSocket
                '-c', '1',                           # Mono
                '-b', '16',                          # 16-bit
                '-e', 'signed-integer',              # PCM
                temp_8k
            ]

            result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)

            # Cleanup temp 24k file immediately
            try:
                os.unlink(temp_24k)
            except:
                pass

            if result.returncode != 0:
                logger.error(f"Sox resampling failed: {result.stderr}")
                return b''

            if not os.path.exists(temp_8k):
                logger.error("Sox output file not created")
                return b''

            # Read WAV file and extract raw PCM data
            audio_8khz, sr = sf.read(temp_8k, dtype='int16')

            # Convert numpy array to bytes
            audio_8khz_bytes = audio_8khz.tobytes()

            file_size = os.path.getsize(temp_8k)
            logger.info(f"Kokoro TTS success: {file_size} bytes WAV, {len(audio_8khz_bytes)} bytes PCM")

            return audio_8khz_bytes

        except Exception as e:
            logger.error(f"Kokoro TTS error: {e}", exc_info=True)
            return b''

        finally:
            # Cleanup temp files
            for temp_file in [temp_24k, temp_8k]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.debug(f"Cleanup failed for {temp_file}: {e}")

    def list_voices(self):
        """List available voices"""
        return list(self.voice_mapping.keys())
