"""
Audio Utilities

Audio processing utilities including resampling.
"""

import struct
import logging

logger = logging.getLogger(__name__)


def resample_8khz_to_16khz(audio_8khz: bytes) -> bytes:
    """
    Resample 8kHz audio to 16kHz using linear interpolation.

    This provides better quality than simple sample duplication.

    Args:
        audio_8khz: 8kHz PCM audio (16-bit signed samples, little-endian)

    Returns:
        16kHz PCM audio (16-bit signed samples, little-endian)
    """
    # Convert bytes to 16-bit signed samples
    num_samples = len(audio_8khz) // 2
    samples_8khz = struct.unpack(f'<{num_samples}h', audio_8khz)

    # Linear interpolation for upsampling
    # For each input sample, we generate 2 output samples
    # The second sample is the average of current and next sample
    samples_16khz = []

    for i in range(num_samples):
        # Original sample
        samples_16khz.append(samples_8khz[i])

        # Interpolated sample (average of current and next)
        if i < num_samples - 1:
            interpolated = (samples_8khz[i] + samples_8khz[i + 1]) // 2
            samples_16khz.append(interpolated)
        else:
            # Last sample: just duplicate
            samples_16khz.append(samples_8khz[i])

    # Convert back to bytes
    return struct.pack(f'<{len(samples_16khz)}h', *samples_16khz)


def resample_16khz_to_8khz(audio_16khz: bytes) -> bytes:
    """
    Downsample 16kHz audio to 8kHz by taking every other sample.

    Args:
        audio_16khz: 16kHz PCM audio (16-bit signed samples, little-endian)

    Returns:
        8kHz PCM audio (16-bit signed samples, little-endian)
    """
    # Convert bytes to 16-bit signed samples
    num_samples = len(audio_16khz) // 2
    samples_16khz = struct.unpack(f'<{num_samples}h', audio_16khz)

    # Take every other sample (simple decimation)
    samples_8khz = samples_16khz[::2]

    # Convert back to bytes
    return struct.pack(f'<{len(samples_8khz)}h', *samples_8khz)
