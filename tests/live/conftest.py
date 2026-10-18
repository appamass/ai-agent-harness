"""Fixtures for the live tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .helpers import OPEN_CLIENTS, OPEN_PROVIDERS, RUN_LIVE_CALLS


@pytest.fixture(autouse=True)
def one_event_loop_per_test() -> Iterator[None]:
    for tracked in (RUN_LIVE_CALLS, OPEN_PROVIDERS, OPEN_CLIENTS):
        tracked.clear()  # nothing may leak from an earlier test: its event loop is gone
    yield
    OPEN_PROVIDERS.clear()
    OPEN_CLIENTS.clear()
    # run_live closes the providers when it ends, so a second call in the same test would use closed clients
    assert len(RUN_LIVE_CALLS) <= 1, "call run_live once per test: put all steps in one async scenario()"
