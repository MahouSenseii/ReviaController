"""
Tests for core/timing.py — pipeline latency measurement.

Covers:
* PipelineTimer lifecycle: begin/start/stop/finish
* TimingRecord population and to_dict() format
* average_ms() — various n values, empty history
* get_summary()
* Persistence: _persist / load_history round-trip, corrupt file, missing file
* pipeline_timing event published on finish()
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.timing as timing_mod
from core.timing import PipelineTimer, TimingRecord


@pytest.fixture
def timer(bus):
    return PipelineTimer(bus)


# ── TimingRecord ──────────────────────────────────────────────────────

class TestTimingRecord:
    def test_to_dict_contains_all_stage_keys(self):
        r = TimingRecord(
            stimulus_ms=1.5, emotion_ms=2.0, decision_ms=0.5,
            metacognition_ms=0.3, inference_ms=200.0, total_ms=210.0,
        )
        d = r.to_dict()
        for key in ("stimulus", "emotion", "decision", "metacognition",
                    "inference", "total"):
            assert key in d, f"Missing key '{key}' in to_dict()"

    def test_to_dict_values_contain_ms_unit(self):
        r = TimingRecord(stimulus_ms=1.5)
        assert "ms" in r.to_dict()["stimulus"]

    def test_to_dict_total_formatted_without_decimal(self):
        r = TimingRecord(total_ms=210.7)
        assert r.to_dict()["total"] == "211 ms"

    def test_to_dict_stimulus_formatted_with_one_decimal(self):
        r = TimingRecord(stimulus_ms=1.5)
        assert r.to_dict()["stimulus"] == "1.5 ms"


# ── PipelineTimer lifecycle ───────────────────────────────────────────

class TestLifecycle:
    def test_full_run_returns_timing_record(self, timer):
        timer.begin()
        timer.start("stimulus")
        timer.stop("stimulus")
        record = timer.finish()
        assert isinstance(record, TimingRecord)

    def test_total_ms_is_non_negative(self, timer):
        timer.begin()
        record = timer.finish()
        assert record.total_ms >= 0.0

    def test_measured_stage_is_non_negative(self, timer):
        timer.begin()
        timer.start("stimulus")
        timer.stop("stimulus")
        record = timer.finish()
        assert record.stimulus_ms >= 0.0

    def test_unmeasured_stage_is_zero(self, timer):
        timer.begin()
        record = timer.finish()
        assert record.stimulus_ms == 0.0

    def test_timestamp_is_positive(self, timer):
        timer.begin()
        record = timer.finish()
        assert record.timestamp > 0.0

    def test_start_before_begin_does_not_crash(self, timer):
        timer.start("stimulus")  # no begin() called first
        timer.stop("stimulus")
        # No exception raised

    def test_stop_unknown_stage_does_not_crash(self, timer):
        timer.begin()
        timer.stop("nonexistent_stage")
        timer.finish()

    def test_multiple_runs_accumulate_in_history(self, timer):
        for _ in range(3):
            timer.begin()
            timer.finish()
        assert timer.last_record() is not None


# ── pipeline_timing event ─────────────────────────────────────────────

class TestTimingEvent:
    def test_event_published_on_finish(self, bus):
        received = []
        bus.subscribe("pipeline_timing", received.append)
        t = PipelineTimer(bus)
        t.begin()
        t.finish()
        assert len(received) == 1

    def test_event_contains_total(self, bus):
        received = []
        bus.subscribe("pipeline_timing", received.append)
        t = PipelineTimer(bus)
        t.begin()
        t.finish()
        assert "total" in received[0]

    def test_event_contains_run_count(self, bus):
        received = []
        bus.subscribe("pipeline_timing", received.append)
        t = PipelineTimer(bus)
        t.begin()
        t.finish()
        assert "runs" in received[0]


# ── average_ms() ──────────────────────────────────────────────────────

class TestAverageMs:
    def test_empty_history_returns_zero(self, timer):
        assert timer.average_ms("total", n=10) == 0.0

    def test_single_run(self, timer):
        timer.begin()
        timer.finish()
        avg = timer.average_ms("total", n=1)
        assert avg >= 0.0

    def test_average_over_multiple_runs(self, timer):
        for _ in range(5):
            timer.begin()
            timer.finish()
        avg = timer.average_ms("total", n=5)
        assert avg >= 0.0

    def test_n_larger_than_history(self, timer):
        timer.begin()
        timer.finish()
        avg = timer.average_ms("total", n=100)
        assert avg >= 0.0

    def test_unknown_stage_returns_zero(self, timer):
        timer.begin()
        timer.finish()
        assert timer.average_ms("nonexistent_stage", n=5) == 0.0


# ── get_summary() ─────────────────────────────────────────────────────

class TestGetSummary:
    def test_no_history_returns_message(self, timer):
        summary = timer.get_summary()
        assert "message" in summary

    def test_with_history_has_last_key(self, timer):
        timer.begin()
        timer.finish()
        assert "last" in timer.get_summary()

    def test_with_history_has_runs(self, timer):
        timer.begin()
        timer.finish()
        assert timer.get_summary()["runs"] == 1


# ── Persistence ───────────────────────────────────────────────────────

class TestPersistence:
    """
    These tests monkey-patch the module-level _HISTORY_FILE so we can
    redirect writes to a tmp_path without touching the real filesystem.
    """

    def test_persist_and_load_round_trip(self, bus, tmp_path):
        original = timing_mod._HISTORY_FILE
        timing_mod._HISTORY_FILE = tmp_path / "timing.json"
        try:
            t = PipelineTimer(bus)
            t.begin()
            t.finish()
            history = t.load_history()
            assert len(history) >= 1
            assert "total_ms" in history[0]
        finally:
            timing_mod._HISTORY_FILE = original

    def test_load_missing_file_returns_empty(self, bus, tmp_path):
        original = timing_mod._HISTORY_FILE
        timing_mod._HISTORY_FILE = tmp_path / "nonexistent.json"
        try:
            t = PipelineTimer(bus)
            assert t.load_history() == []
        finally:
            timing_mod._HISTORY_FILE = original

    def test_load_corrupt_file_returns_empty(self, bus, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json", encoding="utf-8")
        original = timing_mod._HISTORY_FILE
        timing_mod._HISTORY_FILE = bad_path
        try:
            t = PipelineTimer(bus)
            assert t.load_history() == []
        finally:
            timing_mod._HISTORY_FILE = original

    def test_persist_trims_to_max_saved(self, bus, tmp_path):
        original_path = timing_mod._HISTORY_FILE
        original_max = timing_mod._MAX_SAVED
        timing_mod._HISTORY_FILE = tmp_path / "timing.json"
        timing_mod._MAX_SAVED = 3
        try:
            t = PipelineTimer(bus)
            for _ in range(5):
                t.begin()
                t.finish()
            history = t.load_history()
            assert len(history) <= 3
        finally:
            timing_mod._HISTORY_FILE = original_path
            timing_mod._MAX_SAVED = original_max

    def test_persist_does_not_crash_on_bad_path(self, bus):
        original = timing_mod._HISTORY_FILE
        timing_mod._HISTORY_FILE = Path("/nonexistent_dir/timing.json")
        try:
            t = PipelineTimer(bus)
            t.begin()
            t.finish()  # _persist swallows OSError
        finally:
            timing_mod._HISTORY_FILE = original
