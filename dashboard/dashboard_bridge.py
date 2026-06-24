"""
Bridge between the agent (which decides WHAT to do) and the dashboard GUI
(which decides HOW to show it).

Simplest integration: write to a shared JSON file. The forked dashboard
repo (e.g. SihabSahariar/Smart-CAR-Dashboard-GUI-in-Python) polls this file
on a timer in its render loop and updates gauges/AC/nav widgets accordingly.

If you integrate in-process instead (agent and dashboard running in the
same Python process), replace the file read/write below with direct calls
into the dashboard's own state objects.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

STATE_FILE = os.getenv("DASHBOARD_STATE_FILE", "dashboard/dashboard_state.json")

_lock = threading.Lock()


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    state["_updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def update_climate(zone: str, action: str, value: str = "") -> dict:
    with _lock:
        state = _load_state()
        state.setdefault("climate", {})
        state["climate"][zone] = {"action": action, "value": value}
        _save_state(state)
        return state["climate"][zone]


def update_navigation(destination: str) -> dict:
    with _lock:
        state = _load_state()
        state["navigation"] = {"destination": destination}
        _save_state(state)
        return state["navigation"]


def get_state() -> dict:
    with _lock:
        return _load_state()
