/*
 * revia_native.cpp — C++ acceleration for Revia Controller
 *
 * Provides high-performance routines for:
 *   1. Voice Activity Detection (VAD) — energy + zero-crossing
 *   2. Audio DSP — resampling, RMS, spectral features
 *   3. Image preprocessing — grayscale, resize, threshold for OCR
 *   4. Ring buffer for real-time audio streaming
 *
 * Compiled as a shared library (.so) and called from Python via ctypes.
 * No Python.h dependency — pure C ABI for maximum portability.
 */

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>

extern "C" {

/* ─── Audio Ring Buffer ──────────────────────────────────── */

struct AudioRingBuffer {
    float*   buf;
    uint32_t capacity;
    uint32_t write_pos;
    uint32_t read_pos;
    uint32_t count;
};

AudioRingBuffer* ring_create(uint32_t capacity) {
    auto* rb = new AudioRingBuffer;
    rb->buf       = new float[capacity];
    rb->capacity  = capacity;
    rb->write_pos = 0;
    rb->read_pos  = 0;
    rb->count     = 0;
    std::memset(rb->buf, 0, capacity * sizeof(float));
    return rb;
}

void ring_destroy(AudioRingBuffer* rb) {
    if (rb) {
        delete[] rb->buf;
        delete rb;
    }
}

uint32_t ring_write(AudioRingBuffer* rb, const float* data, uint32_t n) {
    uint32_t written = 0;
    for (uint32_t i = 0; i < n && rb->count < rb->capacity; ++i) {
        rb->buf[rb->write_pos] = data[i];
        rb->write_pos = (rb->write_pos + 1) % rb->capacity;
        rb->count++;
        written++;
    }
    return written;
}

uint32_t ring_read(AudioRingBuffer* rb, float* out, uint32_t n) {
    uint32_t read = 0;
    for (uint32_t i = 0; i < n && rb->count > 0; ++i) {
        out[i] = rb->buf[rb->read_pos];
        rb->read_pos = (rb->read_pos + 1) % rb->capacity;
        rb->count--;
        read++;
    }
    return read;
}

uint32_t ring_available(const AudioRingBuffer* rb) {
    return rb->count;
}

void ring_clear(AudioRingBuffer* rb) {
    rb->write_pos = 0;
    rb->read_pos  = 0;
    rb->count     = 0;
}


/* ─── Voice Activity Detection ───────────────────────────── */

/**
 * Compute RMS energy of a float audio buffer.
 */
float audio_rms(const float* samples, uint32_t n) {
    if (n == 0) return 0.0f;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double s = static_cast<double>(samples[i]);
        sum += s * s;
    }
    return static_cast<float>(std::sqrt(sum / n));
}

/**
 * Count zero crossings (sign changes) in buffer.
 * Normalised to per-sample rate.
 */
float audio_zero_crossing_rate(const float* samples, uint32_t n) {
    if (n < 2) return 0.0f;
    uint32_t crossings = 0;
    for (uint32_t i = 1; i < n; ++i) {
        if ((samples[i] >= 0.0f) != (samples[i - 1] >= 0.0f)) {
            crossings++;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(n - 1);
}

/**
 * Compute short-time energy in dB.
 * Returns -100 for silence.
 */
float audio_energy_db(const float* samples, uint32_t n) {
    float rms = audio_rms(samples, n);
    if (rms < 1e-10f) return -100.0f;
    return 20.0f * std::log10(rms);
}

/**
 * Voice Activity Detection using energy + zero-crossing.
 *
 * @param samples       Audio buffer (float, mono)
 * @param n             Number of samples
 * @param energy_thresh Energy threshold in dB (e.g. -35)
 * @param zcr_low       Min ZCR for speech (e.g. 0.02)
 * @param zcr_high      Max ZCR for speech (e.g. 0.30)
 * @return              1 if voice detected, 0 otherwise
 */
int vad_detect(const float* samples, uint32_t n,
               float energy_thresh, float zcr_low, float zcr_high) {
    float energy = audio_energy_db(samples, n);
    if (energy < energy_thresh) return 0;

    float zcr = audio_zero_crossing_rate(samples, n);
    /* Speech typically has moderate ZCR; noise has very high ZCR */
    if (zcr < zcr_low || zcr > zcr_high) return 0;

    return 1;
}

/**
 * Frame-level VAD on an entire buffer.
 * Writes 1/0 per frame into `out_flags`.
 *
 * @param samples        Full audio buffer
 * @param total_samples  Total sample count
 * @param frame_size     Samples per frame (e.g. 480 for 30ms at 16kHz)
 * @param hop_size       Hop between frames
 * @param energy_thresh  dB threshold
 * @param zcr_low        Min ZCR
 * @param zcr_high       Max ZCR
 * @param out_flags      Output array (caller allocates)
 * @return               Number of frames processed
 */
uint32_t vad_detect_frames(const float* samples, uint32_t total_samples,
                           uint32_t frame_size, uint32_t hop_size,
                           float energy_thresh, float zcr_low, float zcr_high,
                           int* out_flags) {
    uint32_t num_frames = 0;
    for (uint32_t offset = 0; offset + frame_size <= total_samples; offset += hop_size) {
        out_flags[num_frames] = vad_detect(
            samples + offset, frame_size,
            energy_thresh, zcr_low, zcr_high
        );
        num_frames++;
    }
    return num_frames;
}

/**
 * Apply a simple pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
 * This boosts high frequencies, improving speech recognition.
 * Operates in-place.
 */
void audio_preemphasis(float* samples, uint32_t n, float coeff) {
    if (n < 2) return;
    /* Process backwards to avoid overwriting needed values */
    for (uint32_t i = n - 1; i >= 1; --i) {
        samples[i] = samples[i] - coeff * samples[i - 1];
    }
    /* First sample: no previous, scale down */
    samples[0] = samples[0] * (1.0f - coeff);
}

/**
 * Simple linear resampling from src_rate to dst_rate.
 * @return Number of output samples written.
 */
uint32_t audio_resample_linear(const float* in, uint32_t in_len,
                               uint32_t src_rate, uint32_t dst_rate,
                               float* out, uint32_t out_capacity) {
    if (in_len == 0 || src_rate == 0 || dst_rate == 0) return 0;

    double ratio = static_cast<double>(src_rate) / static_cast<double>(dst_rate);
    uint32_t out_len = static_cast<uint32_t>(
        std::min(static_cast<double>(out_capacity),
                 std::ceil(static_cast<double>(in_len) / ratio))
    );

    for (uint32_t i = 0; i < out_len; ++i) {
        double src_idx = static_cast<double>(i) * ratio;
        uint32_t idx0 = static_cast<uint32_t>(src_idx);
        uint32_t idx1 = std::min(idx0 + 1, in_len - 1);
        double frac = src_idx - static_cast<double>(idx0);
        out[i] = static_cast<float>(
            static_cast<double>(in[idx0]) * (1.0 - frac) +
            static_cast<double>(in[idx1]) * frac
        );
    }
    return out_len;
}


/* ─── Image Preprocessing (for OCR / Vision) ──────────────── */

/**
 * Convert RGB image to grayscale (luminosity method).
 * in:  RGB packed (3 bytes per pixel, row-major)
 * out: single-channel (1 byte per pixel)
 */
void image_rgb_to_gray(const uint8_t* rgb, uint8_t* gray,
                       uint32_t width, uint32_t height) {
    uint32_t total = width * height;
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t off = i * 3;
        /* ITU-R BT.601 luminosity: 0.299R + 0.587G + 0.114B */
        float lum = 0.299f * rgb[off]
                  + 0.587f * rgb[off + 1]
                  + 0.114f * rgb[off + 2];
        gray[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, lum)));
    }
}

/**
 * Adaptive threshold (mean method) for binarisation.
 * Good for OCR preprocessing.
 *
 * @param gray       Input grayscale image
 * @param out        Output binary image (0 or 255)
 * @param width      Image width
 * @param height     Image height
 * @param block_size Neighbourhood size (must be odd, e.g. 15)
 * @param c          Constant subtracted from mean
 */
void image_adaptive_threshold(const uint8_t* gray, uint8_t* out,
                              uint32_t width, uint32_t height,
                              uint32_t block_size, int c) {
    int half = static_cast<int>(block_size / 2);

    /* Build integral image for O(1) rectangle sums */
    std::vector<int64_t> integral((width + 1) * (height + 1), 0);

    auto IDX = [&](uint32_t y, uint32_t x) -> int64_t& {
        return integral[(y + 1) * (width + 1) + (x + 1)];
    };

    for (uint32_t y = 0; y < height; ++y) {
        int64_t row_sum = 0;
        for (uint32_t x = 0; x < width; ++x) {
            row_sum += gray[y * width + x];
            IDX(y, x) = row_sum + (y > 0 ? IDX(y - 1, x) : 0);
        }
    }

    auto rect_sum = [&](int y0, int x0, int y1, int x1) -> int64_t {
        /* Clamp to bounds */
        x0 = std::max(0, x0);  y0 = std::max(0, y0);
        x1 = std::min(static_cast<int>(width) - 1, x1);
        y1 = std::min(static_cast<int>(height) - 1, y1);
        return integral[(y1 + 1) * (width + 1) + (x1 + 1)]
             - integral[(y0)     * (width + 1) + (x1 + 1)]
             - integral[(y1 + 1) * (width + 1) + (x0)]
             + integral[(y0)     * (width + 1) + (x0)];
    };

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            int x0 = static_cast<int>(x) - half;
            int y0 = static_cast<int>(y) - half;
            int x1 = static_cast<int>(x) + half;
            int y1 = static_cast<int>(y) + half;

            /* Clamp for count */
            int cx0 = std::max(0, x0), cy0 = std::max(0, y0);
            int cx1 = std::min(static_cast<int>(width) - 1, x1);
            int cy1 = std::min(static_cast<int>(height) - 1, y1);
            int count = (cx1 - cx0 + 1) * (cy1 - cy0 + 1);

            int64_t s = rect_sum(y0, x0, y1, x1);
            float mean = static_cast<float>(s) / static_cast<float>(count);

            out[y * width + x] = (gray[y * width + x] > mean - c) ? 255 : 0;
        }
    }
}

/**
 * Bilinear resize of a grayscale image.
 */
void image_resize_bilinear(const uint8_t* src, uint32_t sw, uint32_t sh,
                           uint8_t* dst, uint32_t dw, uint32_t dh) {
    float x_ratio = static_cast<float>(sw - 1) / static_cast<float>(dw);
    float y_ratio = static_cast<float>(sh - 1) / static_cast<float>(dh);

    for (uint32_t dy = 0; dy < dh; ++dy) {
        float sy = dy * y_ratio;
        uint32_t y0 = static_cast<uint32_t>(sy);
        uint32_t y1 = std::min(y0 + 1, sh - 1);
        float fy = sy - y0;

        for (uint32_t dx = 0; dx < dw; ++dx) {
            float sx = dx * x_ratio;
            uint32_t x0 = static_cast<uint32_t>(sx);
            uint32_t x1 = std::min(x0 + 1, sw - 1);
            float fx = sx - x0;

            float val = src[y0 * sw + x0] * (1 - fx) * (1 - fy)
                      + src[y0 * sw + x1] * fx       * (1 - fy)
                      + src[y1 * sw + x0] * (1 - fx) * fy
                      + src[y1 * sw + x1] * fx       * fy;

            dst[dy * dw + dx] = static_cast<uint8_t>(
                std::min(255.0f, std::max(0.0f, val))
            );
        }
    }
}

/**
 * Compute image sharpness (variance of Laplacian).
 * Higher value = sharper image. Useful for blur detection.
 */
float image_sharpness(const uint8_t* gray, uint32_t width, uint32_t height) {
    if (width < 3 || height < 3) return 0.0f;

    double sum = 0.0, sum_sq = 0.0;
    uint32_t count = 0;

    for (uint32_t y = 1; y < height - 1; ++y) {
        for (uint32_t x = 1; x < width - 1; ++x) {
            /* Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]] */
            int lap = -4 * gray[y * width + x]
                    + gray[(y - 1) * width + x]
                    + gray[(y + 1) * width + x]
                    + gray[y * width + (x - 1)]
                    + gray[y * width + (x + 1)];
            double d = static_cast<double>(lap);
            sum    += d;
            sum_sq += d * d;
            count++;
        }
    }

    if (count == 0) return 0.0f;
    double mean = sum / count;
    double variance = sum_sq / count - mean * mean;
    return static_cast<float>(variance);
}


/* ─── Phoneme/Viseme Timing Estimation ─────────────────────── */

/**
 * Estimate phoneme boundaries from text length and speech duration.
 * Simple proportional model — real engines override with actual timestamps.
 *
 * @param char_count      Number of characters in text
 * @param duration_ms     Total speech duration in milliseconds
 * @param out_start_ms    Output: start time per character (caller allocates)
 * @param out_end_ms      Output: end time per character (caller allocates)
 * @param out_count       Output: number of entries written
 */
void estimate_phoneme_timing(uint32_t char_count, float duration_ms,
                             float* out_start_ms, float* out_end_ms,
                             uint32_t* out_count) {
    if (char_count == 0 || duration_ms <= 0.0f) {
        *out_count = 0;
        return;
    }
    float per_char = duration_ms / static_cast<float>(char_count);
    for (uint32_t i = 0; i < char_count; ++i) {
        out_start_ms[i] = per_char * i;
        out_end_ms[i]   = per_char * (i + 1);
    }
    *out_count = char_count;
}


/* ─── Spectral Feature Extraction ──────────────────────────── */

/**
 * Compute spectral centroid of a frame (approximation without FFT).
 * Uses autocorrelation-based pitch period estimation.
 * Returns estimated dominant frequency in Hz given sample_rate.
 */
float audio_spectral_centroid(const float* samples, uint32_t n,
                              uint32_t sample_rate) {
    if (n < 4 || sample_rate == 0) return 0.0f;

    /* Find first zero-crossing period as pitch estimate */
    uint32_t crossings = 0;
    for (uint32_t i = 1; i < n; ++i) {
        if ((samples[i] >= 0.0f) != (samples[i - 1] >= 0.0f)) {
            crossings++;
        }
    }
    if (crossings < 2) return 0.0f;

    /* Each pair of zero crossings = one half period */
    float half_periods = crossings / 2.0f;
    float duration_sec = static_cast<float>(n) / static_cast<float>(sample_rate);
    return half_periods / duration_sec;
}


/* ─── Wake Word Energy Detector ────────────────────────────── */

/**
 * Sliding-window energy detector for potential wake word regions.
 * Finds contiguous voiced segments above threshold.
 *
 * @param samples        Audio buffer
 * @param n              Sample count
 * @param sample_rate    Samples per second
 * @param frame_ms       Frame length in ms
 * @param energy_thresh  dB threshold
 * @param min_dur_ms     Minimum segment duration to be a candidate
 * @param max_dur_ms     Maximum segment duration
 * @param out_starts     Output: segment start sample indices
 * @param out_ends       Output: segment end sample indices
 * @param max_segments   Max output capacity
 * @return               Number of segments found
 */
uint32_t find_voiced_segments(const float* samples, uint32_t n,
                              uint32_t sample_rate,
                              uint32_t frame_ms,
                              float energy_thresh,
                              uint32_t min_dur_ms, uint32_t max_dur_ms,
                              uint32_t* out_starts, uint32_t* out_ends,
                              uint32_t max_segments) {
    uint32_t frame_size = (sample_rate * frame_ms) / 1000;
    if (frame_size == 0 || n < frame_size) return 0;

    uint32_t min_frames = (min_dur_ms + frame_ms - 1) / frame_ms;
    uint32_t max_frames = max_dur_ms / frame_ms;
    uint32_t seg_count = 0;
    uint32_t voiced_run = 0;
    uint32_t run_start = 0;

    for (uint32_t off = 0; off + frame_size <= n; off += frame_size) {
        float e = audio_energy_db(samples + off, frame_size);
        if (e >= energy_thresh) {
            if (voiced_run == 0) run_start = off;
            voiced_run++;
        } else {
            if (voiced_run >= min_frames && voiced_run <= max_frames
                && seg_count < max_segments) {
                out_starts[seg_count] = run_start;
                out_ends[seg_count]   = off;
                seg_count++;
            }
            voiced_run = 0;
        }
    }
    /* Handle trailing segment */
    if (voiced_run >= min_frames && voiced_run <= max_frames
        && seg_count < max_segments) {
        out_starts[seg_count] = run_start;
        out_ends[seg_count]   = n;
        seg_count++;
    }
    return seg_count;
}

} /* extern "C" */
