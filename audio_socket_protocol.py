"""
AudioSocket Protocol Implementation
Handles TLV frame encoding/decoding for Asterisk AudioSocket.
"""

import struct
import asyncio
import logging
import uuid
from enum import IntEnum
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """AudioSocket message types"""
    TERMINATE = 0x00
    UUID = 0x01
    AUDIO = 0x10
    ERROR = 0xFF


class ErrorCode(IntEnum):
    """AudioSocket error codes"""
    HANGUP = 0x01
    FRAME_FORWARDING_ERROR = 0x02
    MEMORY_ERROR = 0x04


@dataclass
class AudioSocketMessage:
    """Represents a parsed AudioSocket message"""
    msg_type: MessageType
    payload: bytes

    @property
    def length(self) -> int:
        return len(self.payload)


class AudioSocketProtocol:
    """
    AudioSocket protocol encoder/decoder.

    Frame format:
    - Type: 1 byte (uint8)
    - Length: 2 bytes (uint16, big-endian)
    - Payload: 0-65535 bytes
    """

    # Protocol constants
    SAMPLE_RATE = 8000
    SAMPLE_WIDTH = 2  # bytes (16-bit)
    CHANNELS = 1
    FRAME_DURATION_MS = 20
    FRAME_SIZE = 320  # bytes (8000 Hz * 0.020 s * 2 bytes = 320)

    HEADER_SIZE = 3  # 1 byte type + 2 bytes length
    MAX_PAYLOAD_SIZE = 65535

    @staticmethod
    def encode_frame(msg_type: MessageType, payload: bytes) -> bytes:
        """
        Encode AudioSocket frame with TLV structure.

        Args:
            msg_type: Message type
            payload: Payload data

        Returns:
            Encoded frame (header + payload)

        Raises:
            ValueError: If payload exceeds 65535 bytes
        """
        payload_len = len(payload)

        if payload_len > AudioSocketProtocol.MAX_PAYLOAD_SIZE:
            raise ValueError(
                f"Payload size {payload_len} exceeds maximum {AudioSocketProtocol.MAX_PAYLOAD_SIZE}"
            )

        # Pack header: Type (1 byte) + Length (2 bytes, big-endian)
        header = struct.pack('>BH', msg_type, payload_len)

        return header + payload

    @staticmethod
    async def decode_frame(reader: asyncio.StreamReader) -> Optional[AudioSocketMessage]:
        """
        Decode AudioSocket frame from TCP stream.

        Args:
            reader: AsyncIO StreamReader

        Returns:
            Decoded message or None if stream closed

        Raises:
            ValueError: If frame format is invalid
        """
        # Read 3-byte header
        header = await reader.readexactly(AudioSocketProtocol.HEADER_SIZE)

        if not header:
            return None  # Stream closed

        # Unpack header
        msg_type, payload_len = struct.unpack('>BH', header)

        # Read payload
        if payload_len > 0:
            payload = await reader.readexactly(payload_len)
        else:
            payload = b''

        return AudioSocketMessage(
            msg_type=MessageType(msg_type),
            payload=payload
        )

    @staticmethod
    def create_uuid_message() -> bytes:
        """Create UUID message for connection handshake."""
        session_uuid = uuid.uuid4()
        return AudioSocketProtocol.encode_frame(
            MessageType.UUID,
            session_uuid.bytes
        )

    @staticmethod
    def create_audio_message(pcm_data: bytes) -> bytes:
        """
        Create audio message from PCM data.

        Args:
            pcm_data: 320 bytes of int16 LE PCM @ 8kHz

        Returns:
            Encoded audio frame

        Raises:
            ValueError: If PCM data size is incorrect
        """
        if len(pcm_data) != AudioSocketProtocol.FRAME_SIZE:
            raise ValueError(
                f"Audio frame must be exactly {AudioSocketProtocol.FRAME_SIZE} bytes, "
                f"got {len(pcm_data)}"
            )

        return AudioSocketProtocol.encode_frame(MessageType.AUDIO, pcm_data)

    @staticmethod
    def create_terminate_message() -> bytes:
        """Create termination message."""
        return AudioSocketProtocol.encode_frame(MessageType.TERMINATE, b'')

    @staticmethod
    def parse_audio_message(message: AudioSocketMessage) -> bytes:
        """Extract PCM data from audio message."""
        if message.msg_type != MessageType.AUDIO:
            raise ValueError(f"Expected AUDIO message, got {message.msg_type}")

        return message.payload
