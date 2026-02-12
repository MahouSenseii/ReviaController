"""
core/native_bridge.py — ctypes bridge to librevia_native.so

Provides Python wrappers around the C++ acceleration library.
Falls back to pure-Python implementations if the .so is missing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import array
from pathlib import Path
from typing import List, Tuple, Optional

# ── Load native library ─────────────────────────────────────

_LIB: Optional[ctypes.CDLL] = None
_LIB_PATHS = [
    Path(__file__).resolve().parent.parent / "librevia_native.so",
    Path(os.getcwd()) / "librevia_native.so",
]

for _p in _LIB_PATHS:
    if _p.exists():
        try:
            _LIB = ctypes.CDLL(str(_p))
            break
        except OSError:
            continue

HAS_NATIVE = _LIB is not None


# ── Type helpers ─────────────────────────────────────────────

_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
_c_int_p   = ctypes.POINTER(ctypes.c_int)
_c_uint_p  = ctypes.POINTER(ctypes.c_uint32)


def _float_array(data: List[float]) -> ctypes.Array:
    """Create a ctypes float array from a Python list."""
    arr = (ctypes.c_float * len(data))(*data)
    return arr


def _float_buf(size: int) -> ctypes.Array:
    """Allocate a zeroed float buffer."""
    return (ctypes.c_float * size)()


def _int_buf(size: int) -> ctypes.Array:
    return (ctypes.c_int * size)()


def _uint_buf(size: int) -> ctypes.Array:
    return (ctypes.c_uint32 * size)()


def _uint8_buf(size: int) -> ctypes.Array:
    return (ctypes.c_uint8 * size)()


# ── Setup function signatures ────────────────────────────────

if _LIB is not None:
    # Audio
    _LIB.audio_rms.argtypes = [_c_float_p, ctypes.c_uint32]
    _LIB.audio_rms.restype  = ctypes.c_float

    _LIB.audio_zero_crossing_rate.argtypes = [_c_float_p, ctypes.c_uint32]
    _LIB.audio_zero_crossing_rate.restype  = ctypes.c_float

    _LIB.audio_energy_db.argtypes = [_c_float_p, ctypes.c_uint32]
    _LIB.audio_energy_db.restype  = ctypes.c_float

    _LIB.vad_detect.argtypes = [
        _c_float_p, ctypes.c_uint32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _LIB.vad_detect.restype = ctypes.c_int

    _LIB.vad_detect_frames.argtypes = [
        _c_float_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        _c_int_p,
    ]
    _LIB.vad_detect_frames.restype = ctypes.c_uint32

    _LIB.audio_preemphasis.argtypes = [_c_float_p, ctypes.c_uint32, ctypes.c_float]
    _LIB.audio_preemphasis.restype  = None

    _LIB.audio_resample_linear.argtypes = [
        _c_float_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32,
        _c_float_p, ctypes.c_uint32,
    ]
    _LIB.audio_resample_linear.restype = ctypes.c_uint32

    _LIB.audio_spectral_centroid.argtypes = [_c_float_p, ctypes.c_uint32, ctypes.c_uint32]
    _LIB.audio_spectral_centroid.restype  = ctypes.c_float

    # Image
    _LIB.image_rgb_to_gray.argtypes = [_c_uint8_p, _c_uint8_p, ctypes.c_uint32, ctypes.c_uint32]
    _LIB.image_rgb_to_gray.restype  = None

    _LIB.image_adaptive_threshold.argtypes = [
        _c_uint8_p, _c_uint8_p,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_int,
    ]
    _LIB.image_adaptive_threshold.restype = None

    _LIB.image_resize_bilinear.argtypes = [
        _c_uint8_p, ctypes.c_uint32, ctypes.c_uint32,
        _c_uint8_p, ctypes.c_uint32, ctypes.c_uint32,
    ]
    _LIB.image_resize_bilinear.restype = None

    _LIB.image_sharpness.argtypes = [_c_uint8_p, ctypes.c_uint32, ctypes.c_uint32]
    _LIB.image_sharpness.restype  = ctypes.c_float

    _LIB.estimate_phoneme_timing.argtypes = [
        ctypes.c_uint32, ctypes.c_float,
        _c_float_p, _c_float_p, _c_uint_p,
    ]
    _LIB.estimate_phoneme_timing.restype = None

    _LIB.find_voiced_segments.argtypes = [
        _c_float_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32, ctypes.c_uint32,
        _c_uint_p, _c_uint_p, ctypes.c_uint32,
    ]
    _LIB.find_voiced_segments.restype = ctypes.c_uint32

    # Ring buffer
    _LIB.ring_create.argtypes  = [ctypes.c_uint32]
    _LIB.ring_create.restype   = ctypes.c_void_p

    _LIB.ring_destroy.argtypes = [ctypes.c_void_p]
    _LIB.ring_destroy.restype  = None

    _LIB.ring_write.argtypes   = [ctypes.c_void_p, _c_float_p, ctypes.c_uint32]
    _LIB.ring_write.restype    = ctypes.c_uint32

    _LIB.ring_read.argtypes    = [ctypes.c_void_p, _c_float_p, ctypes.c_uint32]
    _LIB.ring_read.restype     = ctypes.c_uint32

    _LIB.ring_available.argtypes = [ctypes.c_void_p]
    _LIB.ring_available.restype  = ctypes.c_uint32

    _LIB.ring_clear.argtypes = [ctypes.c_void_p]
    _LIB.ring_clear.restype  = None


# ── Python API ───────────────────────────────────────────────

def audio_rms(samples: List[float]) -> float:
    """Compute RMS energy of audio samples."""
    if _LIB and len(samples) > 0:
        arr = _float_array(samples)
        return float(_LIB.audio_rms(arr, len(samples)))
    # Fallback
    if not samples:
        return 0.0
    s = sum(x * x for x in samples)
    return (s / len(samples)) ** 0.5


def audio_energy_db(samples: List[float]) -> float:
    """Compute energy in dB."""
    if _LIB and len(samples) > 0:
        arr = _float_array(samples)
        return float(_LIB.audio_energy_db(arr, len(samples)))
    rms = audio_rms(samples)
    if rms < 1e-10:
        return -100.0
    import math
    return 20.0 * math.log10(rms)


def vad_detect(samples: List[float],
               energy_thresh: float = -35.0,
               zcr_low: float = 0.02,
               zcr_high: float = 0.30) -> bool:
    """Detect voice activity in a single frame."""
    if _LIB and len(samples) > 0:
        arr = _float_array(samples)
        return bool(_LIB.vad_detect(arr, len(samples), energy_thresh, zcr_low, zcr_high))
    # Fallback
    e = audio_energy_db(samples)
    if e < energy_thresh:
        return False
    if len(samples) < 2:
        return False
    crossings = sum(
        1 for i in range(1, len(samples))
        if (samples[i] >= 0) != (samples[i - 1] >= 0)
    )
    zcr = crossings / (len(samples) - 1)
    return zcr_low <= zcr <= zcr_high


def vad_detect_frames(samples: List[float],
                      frame_size: int = 480,
                      hop_size: int = 480,
                      energy_thresh: float = -35.0,
                      zcr_low: float = 0.02,
                      zcr_high: float = 0.30) -> List[bool]:
    """Frame-level VAD on entire buffer."""
    n = len(samples)
    if n < frame_size:
        return []
    max_frames = (n - frame_size) // hop_size + 1
    if _LIB:
        arr = _float_array(samples)
        flags = _int_buf(max_frames)
        nf = _LIB.vad_detect_frames(
            arr, n, frame_size, hop_size,
            energy_thresh, zcr_low, zcr_high, flags,
        )
        return [bool(flags[i]) for i in range(nf)]
    # Fallback
    result = []
    for off in range(0, n - frame_size + 1, hop_size):
        result.append(vad_detect(samples[off:off + frame_size],
                                 energy_thresh, zcr_low, zcr_high))
    return result


def audio_preemphasis(samples: List[float], coeff: float = 0.97) -> List[float]:
    """Apply pre-emphasis filter (in-place via copy)."""
    if not samples:
        return []
    if _LIB:
        arr = _float_array(samples)
        _LIB.audio_preemphasis(arr, len(samples), coeff)
        return [arr[i] for i in range(len(samples))]
    out = list(samples)
    for i in range(len(out) - 1, 0, -1):
        out[i] = out[i] - coeff * out[i - 1]
    out[0] = out[0] * (1.0 - coeff)
    return out


def audio_resample(samples: List[float], src_rate: int, dst_rate: int) -> List[float]:
    """Linear resampling."""
    if src_rate == dst_rate:
        return list(samples)
    import math
    out_len = int(math.ceil(len(samples) * dst_rate / src_rate))
    if _LIB and len(samples) > 0:
        arr = _float_array(samples)
        out = _float_buf(out_len)
        n = _LIB.audio_resample_linear(arr, len(samples), src_rate, dst_rate, out, out_len)
        return [out[i] for i in range(n)]
    # Fallback
    ratio = src_rate / dst_rate
    result = []
    for i in range(out_len):
        src_idx = i * ratio
        idx0 = int(src_idx)
        idx1 = min(idx0 + 1, len(samples) - 1)
        frac = src_idx - idx0
        result.append(samples[idx0] * (1 - frac) + samples[idx1] * frac)
    return result


def image_to_gray(rgb_bytes: bytes, width: int, height: int) -> bytes:
    """Convert RGB image to grayscale."""
    if _LIB:
        rgb = (ctypes.c_uint8 * len(rgb_bytes)).from_buffer_copy(rgb_bytes)
        gray = _uint8_buf(width * height)
        _LIB.image_rgb_to_gray(rgb, gray, width, height)
        return bytes(gray)
    # Fallback
    out = bytearray(width * height)
    for i in range(width * height):
        off = i * 3
        lum = 0.299 * rgb_bytes[off] + 0.587 * rgb_bytes[off + 1] + 0.114 * rgb_bytes[off + 2]
        out[i] = min(255, max(0, int(lum)))
    return bytes(out)


def image_adaptive_threshold(gray_bytes: bytes, width: int, height: int,
                             block_size: int = 15, c: int = 5) -> bytes:
    """Adaptive threshold for OCR preprocessing."""
    if _LIB:
        gray = (ctypes.c_uint8 * len(gray_bytes)).from_buffer_copy(gray_bytes)
        out = _uint8_buf(width * height)
        _LIB.image_adaptive_threshold(gray, out, width, height, block_size, c)
        return bytes(out)
    # Fallback — simple global threshold
    mean_val = sum(gray_bytes) / len(gray_bytes) if gray_bytes else 128
    out = bytearray(len(gray_bytes))
    for i, v in enumerate(gray_bytes):
        out[i] = 255 if v > mean_val - c else 0
    return bytes(out)


def image_sharpness(gray_bytes: bytes, width: int, height: int) -> float:
    """Compute image sharpness (variance of Laplacian)."""
    if _LIB and width >= 3 and height >= 3:
        gray = (ctypes.c_uint8 * len(gray_bytes)).from_buffer_copy(gray_bytes)
        return float(_LIB.image_sharpness(gray, width, height))
    return 0.0


def estimate_phoneme_timing(char_count: int,
                            duration_ms: float) -> List[Tuple[float, float]]:
    """Estimate phoneme start/end times from text length and duration."""
    if char_count == 0 or duration_ms <= 0:
        return []
    if _LIB:
        starts = _float_buf(char_count)
        ends   = _float_buf(char_count)
        count  = _uint_buf(1)
        _LIB.estimate_phoneme_timing(char_count, duration_ms, starts, ends, count)
        n = count[0]
        return [(starts[i], ends[i]) for i in range(n)]
    # Fallback
    per = duration_ms / char_count
    return [(per * i, per * (i + 1)) for i in range(char_count)]


def find_voiced_segments(samples: List[float],
                         sample_rate: int = 16000,
                         frame_ms: int = 30,
                         energy_thresh: float = -35.0,
                         min_dur_ms: int = 200,
                         max_dur_ms: int = 3000) -> List[Tuple[int, int]]:
    """Find contiguous voiced regions in audio."""
    if not samples:
        return []
    max_seg = 64
    if _LIB:
        arr = _float_array(samples)
        starts = _uint_buf(max_seg)
        ends   = _uint_buf(max_seg)
        n = _LIB.find_voiced_segments(
            arr, len(samples), sample_rate, frame_ms,
            energy_thresh, min_dur_ms, max_dur_ms,
            starts, ends, max_seg,
        )
        return [(starts[i], ends[i]) for i in range(n)]
    # Fallback
    frame_size = sample_rate * frame_ms // 1000
    if frame_size == 0:
        return []
    min_frames = (min_dur_ms + frame_ms - 1) // frame_ms
    max_frames = max_dur_ms // frame_ms
    segments = []
    run = 0
    run_start = 0
    for off in range(0, len(samples) - frame_size + 1, frame_size):
        e = audio_energy_db(samples[off:off + frame_size])
        if e >= energy_thresh:
            if run == 0:
                run_start = off
            run += 1
        else:
            if min_frames <= run <= max_frames:
                segments.append((run_start, off))
            run = 0
    if min_frames <= run <= max_frames:
        segments.append((run_start, len(samples)))
    return segments


class NativeRingBuffer:
    """Python wrapper around the C ring buffer (or pure-Python fallback)."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        if _LIB:
            self._handle = _LIB.ring_create(capacity)
            self._native = True
        else:
            self._buf: List[float] = []
            self._native = False

    def write(self, data: List[float]) -> int:
        if self._native:
            arr = _float_array(data)
            return int(_LIB.ring_write(self._handle, arr, len(data)))
        space = self._capacity - len(self._buf)
        n = min(len(data), space)
        self._buf.extend(data[:n])
        return n

    def read(self, n: int) -> List[float]:
        if self._native:
            out = _float_buf(n)
            got = _LIB.ring_read(self._handle, out, n)
            return [out[i] for i in range(got)]
        got = self._buf[:n]
        self._buf = self._buf[n:]
        return got

    def available(self) -> int:
        if self._native:
            return int(_LIB.ring_available(self._handle))
        return len(self._buf)

    def clear(self):
        if self._native:
            _LIB.ring_clear(self._handle)
        else:
            self._buf.clear()

    def __del__(self):
        if self._native and hasattr(self, '_handle'):
            _LIB.ring_destroy(self._handle)
