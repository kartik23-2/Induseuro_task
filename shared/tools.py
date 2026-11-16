from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def get_order(order_id: str) -> dict:
    path = DATA / "orders" / f"{order_id}.json"
    return json.loads(path.read_text())


def get_ticket(ticket_id: str) -> dict:
    path = DATA / "tickets" / f"{ticket_id}.json"
    return json.loads(path.read_text())


def format_thread(ticket: dict) -> str:
    lines = []
    for msg in ticket["messages"]:
        role = msg["from"].upper()
        lines.append(f"[{role}] {msg['text']}")
    return "\n".join(lines)
