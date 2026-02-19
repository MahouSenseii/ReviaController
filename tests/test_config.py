"""
Tests for core/config.py — JSON-backed hierarchical configuration store.

Covers:
* get() — single-level, nested, missing keys, non-dict intermediaries
* set() — flat, nested, overwrite, save=False behaviour
* Persistence — round-trip load/save, corrupt file recovery
* section() — subtree extraction, shallow-copy semantics
* Events — config_changed published on every set()
"""

from __future__ import annotations

import pytest

from core.config import Config


@pytest.fixture
def cfg(tmp_path, bus):
    """Fresh Config backed by a temp file for each test."""
    return Config(bus, path=tmp_path / "config.json")


# ── get() ─────────────────────────────────────────────────────────────

class TestGet:
    def test_missing_key_returns_none(self, cfg):
        assert cfg.get("nonexistent") is None

    def test_missing_key_custom_default(self, cfg):
        assert cfg.get("nonexistent", 42) == 42

    def test_flat_key_after_set(self, cfg):
        cfg.set("foo", "bar", save=False)
        assert cfg.get("foo") == "bar"

    def test_nested_key(self, cfg):
        cfg.set("a.b.c", 99, save=False)
        assert cfg.get("a.b.c") == 99

    def test_nested_missing_leaf(self, cfg):
        cfg.set("a.b", 1, save=False)
        assert cfg.get("a.b.c", "default") == "default"

    def test_intermediate_non_dict_returns_default(self, cfg):
        cfg.set("a", "string", save=False)
        assert cfg.get("a.b", "fallback") == "fallback"

    def test_boolean_value(self, cfg):
        cfg.set("flag", False, save=False)
        assert cfg.get("flag") is False

    def test_list_value(self, cfg):
        cfg.set("items", [1, 2, 3], save=False)
        assert cfg.get("items") == [1, 2, 3]


# ── set() ─────────────────────────────────────────────────────────────

class TestSet:
    def test_flat_set_and_get(self, cfg):
        cfg.set("k", "v", save=False)
        assert cfg.get("k") == "v"

    def test_nested_set(self, cfg):
        cfg.set("x.y.z", 7, save=False)
        assert cfg.get("x.y.z") == 7

    def test_overwrite_existing(self, cfg):
        cfg.set("k", 1, save=False)
        cfg.set("k", 2, save=False)
        assert cfg.get("k") == 2

    def test_sibling_keys_independent(self, cfg):
        cfg.set("ns.a", 1, save=False)
        cfg.set("ns.b", 2, save=False)
        assert cfg.get("ns.a") == 1
        assert cfg.get("ns.b") == 2

    def test_save_false_does_not_write_file(self, tmp_path, bus):
        path = tmp_path / "cfg.json"
        c = Config(bus, path=path)
        c.set("x", 1, save=False)
        assert not path.exists()

    def test_save_true_writes_file(self, tmp_path, bus):
        path = tmp_path / "cfg.json"
        c = Config(bus, path=path)
        c.set("x", 1, save=True)
        assert path.exists()

    def test_publishes_config_changed_event(self, bus, tmp_path):
        c = Config(bus, path=tmp_path / "cfg.json")
        received = []
        bus.subscribe("config_changed", received.append)
        c.set("key", "value", save=False)
        assert len(received) == 1
        assert received[0]["key"] == "key"
        assert received[0]["value"] == "value"

    def test_event_published_even_with_save_false(self, bus, tmp_path):
        c = Config(bus, path=tmp_path / "cfg.json")
        received = []
        bus.subscribe("config_changed", received.append)
        c.set("k", 99, save=False)
        assert len(received) == 1


# ── Persistence ───────────────────────────────────────────────────────

class TestPersistence:
    def test_round_trip(self, tmp_path, bus):
        path = tmp_path / "cfg.json"
        c1 = Config(bus, path=path)
        c1.set("section.key", "hello")

        c2 = Config(bus, path=path)
        assert c2.get("section.key") == "hello"

    def test_multiple_keys_persisted(self, tmp_path, bus):
        path = tmp_path / "cfg.json"
        c1 = Config(bus, path=path)
        c1.set("a", 1)
        c1.set("b", 2)

        c2 = Config(bus, path=path)
        assert c2.get("a") == 1
        assert c2.get("b") == 2

    def test_missing_file_starts_empty(self, tmp_path, bus):
        c = Config(bus, path=tmp_path / "nonexistent.json")
        assert c.get("anything") is None

    def test_corrupt_json_resets_to_empty(self, tmp_path, bus):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json}", encoding="utf-8")
        c = Config(bus, path=path)
        assert c.get("anything") is None


# ── section() ─────────────────────────────────────────────────────────

class TestSection:
    def test_returns_correct_subtree(self, cfg):
        cfg.set("ns.a", 1, save=False)
        cfg.set("ns.b", 2, save=False)
        assert cfg.section("ns") == {"a": 1, "b": 2}

    def test_is_shallow_copy(self, cfg):
        cfg.set("ns.val", 10, save=False)
        sec = cfg.section("ns")
        sec["val"] = 999
        assert cfg.get("ns.val") == 10  # original untouched

    def test_missing_section_returns_empty_dict(self, cfg):
        assert cfg.section("does_not_exist") == {}

    def test_non_dict_section_returns_empty_dict(self, cfg):
        cfg.set("flat", "scalar", save=False)
        assert cfg.section("flat") == {}

    def test_nested_section(self, cfg):
        cfg.set("a.b.x", 1, save=False)
        cfg.set("a.b.y", 2, save=False)
        assert cfg.section("a.b") == {"x": 1, "y": 2}
