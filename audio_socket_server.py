"""
AudioSocket TCP Server
Handles incoming AudioSocket connections from Asterisk.
"""

import asyncio
import logging
import socket
from typing import Optional, Callable
from audio_socket_protocol import (
    AudioSocketProtocol, AudioSocketMessage, MessageType, ErrorCode
)

logger = logging.getLogger(__name__)


class AudioSocketConnection:
    """
    Manages a single AudioSocket TCP connection.

    Provides bidirectional audio streaming with proper error handling.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        on_audio_received: Optional[Callable[[bytes], None]] = None
    ):
        self.reader = reader
        self.writer = writer
        self.on_audio_received = on_audio_received

        self.active = False
        self.session_uuid: Optional[bytes] = None

        # Get peer address
        peername = writer.get_extra_info('peername')
        self.peer_address = f"{peername[0]}:{peername[1]}" if peername else "unknown"

        logger.info(f"AudioSocket connection from {self.peer_address}")

    async def start(self):
        """Start connection and send UUID handshake."""
        try:
            # Enable TCP_NODELAY (critical for low latency)
            sock = self.writer.get_extra_info('socket')
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.debug("TCP_NODELAY enabled")

            # Send UUID message
            uuid_msg = AudioSocketProtocol.create_uuid_message()
            self.writer.write(uuid_msg)
            await self.writer.drain()

            self.active = True
            logger.info(f"AudioSocket connection established: {self.peer_address}")

        except Exception as e:
            logger.error(f"Failed to start AudioSocket connection: {e}")
            self.active = False
            raise

    async def receive_audio(self):
        """
        Receive audio frames from Asterisk (user speech).

        This is the inbound stream - runs continuously in background.
        """
        try:
            while self.active:
                # Decode frame
                message = await AudioSocketProtocol.decode_frame(self.reader)

                if message is None:
                    logger.info("AudioSocket stream closed by peer")
                    break

                # Handle message types
                if message.msg_type == MessageType.AUDIO:
                    # Extract PCM data
                    pcm_data = AudioSocketProtocol.parse_audio_message(message)

                    # Call audio callback
                    if self.on_audio_received:
                        self.on_audio_received(pcm_data)

                elif message.msg_type == MessageType.TERMINATE:
                    logger.info("Received TERMINATE message")
                    break

                elif message.msg_type == MessageType.ERROR:
                    error_code = message.payload[0] if message.payload else 0
                    logger.warning(f"Received ERROR: code={error_code}")
                    if error_code == ErrorCode.HANGUP:
                        logger.info("Call hung up")
                        break

                else:
                    logger.warning(f"Unknown message type: {message.msg_type}")

        except asyncio.IncompleteReadError:
            logger.info("AudioSocket connection closed (incomplete read)")
        except Exception as e:
            logger.error(f"Error receiving audio: {e}", exc_info=True)
        finally:
            self.active = False

    async def send_audio(self, pcm_data: bytes):
        """
        Send audio frame to Asterisk (bot speech).

        Args:
            pcm_data: 320 bytes of int16 LE PCM @ 8kHz
        """
        if not self.active:
            logger.warning("Cannot send audio - connection not active")
            return

        try:
            # Create and send audio message
            audio_msg = AudioSocketProtocol.create_audio_message(pcm_data)
            self.writer.write(audio_msg)
            await self.writer.drain()

        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            self.active = False

    async def close(self):
        """Close connection gracefully."""
        try:
            if self.active:
                # Send TERMINATE message
                terminate_msg = AudioSocketProtocol.create_terminate_message()
                self.writer.write(terminate_msg)
                await self.writer.drain()

            self.writer.close()
            await self.writer.wait_closed()
            logger.info(f"AudioSocket connection closed: {self.peer_address}")

        except Exception as e:
            logger.error(f"Error closing connection: {e}")
        finally:
            self.active = False


class AudioSocketServer:
    """
    AudioSocket TCP server.

    Listens for incoming connections from Asterisk AudioSocket application.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9092,
        on_connection: Optional[Callable[[AudioSocketConnection], None]] = None
    ):
        self.host = host
        self.port = port
        self.on_connection = on_connection

        self.server: Optional[asyncio.Server] = None
        self.active_connections: list[AudioSocketConnection] = []

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle incoming AudioSocket connection."""
        connection = None
        try:
            # Create connection handler
            connection = AudioSocketConnection(reader, writer)
            self.active_connections.append(connection)

            # Start connection (send UUID)
            await connection.start()

            # Notify callback
            if self.on_connection:
                self.on_connection(connection)

            # Start receiving audio
            await connection.receive_audio()

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)

        finally:
            if connection:
                await connection.close()
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

    async def start(self):
        """Start AudioSocket server."""
        self.server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )

        addr = self.server.sockets[0].getsockname()
        logger.info(f"AudioSocket server listening on {addr[0]}:{addr[1]}")

        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop AudioSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("AudioSocket server stopped")
