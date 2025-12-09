"""AudioSocket Voicebot Configuration"""

import os
import sys
from pathlib import Path
from enum import Enum

# Add parent directory to path to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig

# Temp directory for audio files
TEMP_DIR = PathsConfig.TEMP_DIR


class AudioSocketConfig:
    """AudioSocket TCP server configuration"""
    HOST = "127.0.0.1"  # Listen address
    PORT = 9092         # TCP port for AudioSocket

    # Audio format (matches AudioSocket protocol)
    SAMPLE_RATE = 8000  # Hz (telephony standard)
    SAMPLE_WIDTH = 2    # bytes (16-bit)
    CHANNELS = 1        # mono
    FRAME_SIZE = 320    # bytes (20ms @ 8kHz)
    FRAME_DURATION_MS = 20


class AudioConfig:
    """Audio processing configuration - EASY TO TUNE"""

    # ===== Vosk ASR Configuration =====
    VOSK_SAMPLE_RATE = 16000
    VOSK_MODEL_PATH = PathsConfig.VOSK_MODEL_PATH

    # ===== VAD Configuration - TUNE THESE =====
    VAD_SAMPLE_RATE = 8000           # Matches AudioSocket (don't change)
    VAD_FRAME_DURATION_MS = 20       # Frame size in milliseconds (don't change)
    VAD_AGGRESSIVENESS = 1           # 0-3 (0=least sensitive, 3=most sensitive)
                                      # CHANGED from 0 to 1 for better phone line performance
                                      # Increase if VAD misses speech
                                      # Decrease if VAD detects too much noise

    # ===== VAD Hysteresis (prevents flickering) - TUNE THESE =====
    VAD_SPEECH_START_FRAMES = 8      # Frames of speech needed to start detection (8 = 160ms)
                                      # Increase to prevent backchanneling ("uh-huh") from triggering
                                      # Decrease for faster speech detection

    VAD_SPEECH_END_FRAMES = 20       # Frames of silence needed to end detection (20 = 400ms)
                                      # Increase to allow longer pauses within speech
                                      # Decrease to detect end of speech faster

    # ===== Speech Detection Timeouts - TUNE THESE =====
    SILENCE_FRAMES_TO_END_SPEECH = 25  # 500ms silence to end speech (25 frames @ 20ms)
                                        # Increase for slower speakers
                                        # Decrease for faster response

    MAX_SPEECH_FRAMES = 1000            # 20 seconds max speech (1000 frames @ 20ms)
                                         # Increase to allow longer user speech
                                         # Decrease for faster cutoff

    MIN_SPEECH_BYTES = 1600             # 100ms minimum speech required (1600 bytes @ 8kHz)
                                         # Increase to filter out very short sounds


class KokoroConfig:
    """Kokoro TTS Configuration - EASY TO TUNE"""
    VOICE = "af_heart"           # Default voice (af_heart, af_sky, af_bella, af_nova, etc.)
    SPEED = 0.92                 # Speech speed (0.8-1.2, lower=slower, higher=faster)
    SAMPLE_RATE = 24000          # Kokoro native rate (don't change)
    TARGET_SAMPLE_RATE = 8000    # AudioSocket rate (don't change)


class TurnTakingConfig:
    """Turn-taking and interruption configuration - TUNE THESE"""
    # ===== Interruption Detection =====
    INTERRUPTION_ENABLED = True                   # Enable/disable interruption
    INTERRUPTION_ENERGY_THRESHOLD = 300          # RMS energy to detect interruption
                                                  # Increase if false interruptions
                                                  # Decrease if not detecting interruptions

    INTERRUPTION_CONSECUTIVE_FRAMES = 6          # 120ms of speech to confirm interruption (6 frames @ 20ms)
                                                  # INCREASED from 3 to reduce false interruptions from echo/noise
                                                  # Increase to reduce false interruptions
                                                  # Decrease for faster interruption response

    # ===== VAD Pre-filtering =====
    VAD_ENERGY_THRESHOLD = 150                   # RMS energy to filter channel noise before VAD
                                                  # INCREASED from 50 to 150 for phone line quality
                                                  # Increase if detecting too much background noise
                                                  # Decrease if missing quiet speech


class LLMConfig:
    """LLM configuration (REUSED)"""
    MODEL_NAME = "qwen2.5:3b"  # Temporarily using old model until qwen2.5:3b is downloaded
                               # TODO: Download on server: ollama pull qwen2.5:3b
                               # Then change to: MODEL_NAME = "qwen2.5:3b"
    BASE_URL = "http://localhost:11434"
    TIMEOUT = 30


class RAGConfig:
    """RAG (Retrieval-Augmented Generation) Configuration"""

    # ===== Enable/Disable RAG =====
    ENABLED = True  # Set to False to disable RAG globally

    # ===== ChromaDB Path Configuration =====
    # ChromaDB stores ALL customers in a SINGLE database at ./chroma_db/
    # Each customer is a separate collection within this shared database
    CHROMA_DB_PATH = str(PathsConfig.CHROMA_DB_PATH)

    # ===== Embedding Model Configuration =====
    # CRITICAL: Must match the model used during indexing!
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # DO NOT CHANGE unless you re-index
    EXPECTED_DIMENSIONS = 768  # Embedding dimensions for validation

    # ===== Retrieval Settings - TUNE THESE =====
    TOP_K = 3  # Number of document chunks to retrieve
                # Increase for more context (slower, more tokens)
                # Decrease for faster responses (less context)

    MIN_SIMILARITY_SCORE = 0.3  # Minimum relevance score (0-1)
                                 # Increase to filter out irrelevant chunks
                                 # Decrease to retrieve more candidates

    # ===== Context Formatting - TUNE THESE =====
    MAX_CONTEXT_LENGTH = 2000  # Maximum characters for retrieved context
                                # Increase if LLM has larger context window
                                # Decrease to save tokens

    INCLUDE_SOURCE_METADATA = True  # Include doc name, page numbers in context
                                     # Useful for debugging, but adds tokens

    # ===== Performance Settings =====
    LAZY_LOAD = True  # Only initialize ChromaDB when customer_id is provided
                      # Keeps startup fast for non-RAG calls

    CACHE_EMBEDDINGS = False  # Cache query embeddings (experimental)
                              # May improve latency for repeated queries

    # ===== Error Handling =====
    FALLBACK_ON_ERROR = True  # Fall back to LLM-only mode on RAG errors
                              # Set to False to raise errors (for debugging)

    LOG_RETRIEVAL = True  # Log all retrieval attempts (verbose)
                          # Useful for debugging, disable in production if logs are too large


class ZabbixConfig:
    """Zabbix alert configuration"""
    # Alert server URL (where alert data is fetched)
    ALERT_SERVER_URL = "http://localhost:5000"  # Update with actual alert server

    # Alert call identification
    ALERT_CALL_ID_PREFIX = "zabbix_alert_"  # Prefix for alert call UUIDs

    # DTMF configuration for alert acknowledgment
    DTMF_SAMPLE_RATE = 8000  # Sample rate for DTMF detection
    DTMF_FRAME_DURATION_MS = 20  # Frame duration in ms
    DTMF_ENERGY_THRESHOLD = 100  # Energy threshold for DTMF detection
    DTMF_TONE_THRESHOLD = 0.3  # Tone detection threshold
    DTMF_MIN_DURATION_MS = 80  # Minimum DTMF tone duration
    DTMF_WAIT_TIMEOUT = 30  # Seconds to wait for DTMF response


class ConversationState(str, Enum):
    """Conversation states (REUSED)"""
    IDLE = "IDLE"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AI_SPEAKING = "AI_SPEAKING"
