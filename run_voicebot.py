#!/usr/bin/env python3
"""
AudioSocket Voicebot Runner
Start the voicebot server or run tests.
"""

import sys
import subprocess
from pathlib import Path


def run_protocol_tests():
    """Run protocol tests"""
    print("=" * 60)
    print("Running AudioSocket Protocol Tests")
    print("=" * 60)
    subprocess.run([sys.executable, "test_protocol.py"])


def run_voicebot():
    """Start the voicebot server"""
    print("=" * 60)
    print("Starting AudioSocket Voicebot Server")
    print("=" * 60)
    print("Server will listen on 127.0.0.1:9092")
    print("Press Ctrl+C to stop")
    print()
    subprocess.run([sys.executable, "audio_socket_voicebot.py"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_protocol_tests()
    else:
        run_voicebot()


if __name__ == "__main__":
    main()
