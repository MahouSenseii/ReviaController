"""
Shared pytest fixtures.

A session-scoped QCoreApplication is required for any test that creates
a PyQt6 object (EventBus, Config, StimulusAnalyser, etc.).  We use
QCoreApplication (not QApplication) and force the offscreen platform so
the tests run headlessly on CI / servers without a display.
"""

from __future__ import annotations

import os
import sys
import pytest

# Force Qt to run without a display before any Qt import happens.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """One QCoreApplication for the entire test session."""
    from PyQt6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    yield app


@pytest.fixture
def bus(qapp):
    """Fresh EventBus for each test."""
    from core.events import EventBus

    return EventBus()
