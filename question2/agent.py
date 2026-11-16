from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """Same product as Question 1 — triage, retrieve, compose, audit."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def _is_denial_or_escalation(self, order: dict) -> bool:
        ptype = order.get("product_type")
        if ptype == "subscription":
            return int(order.get("subscription_age_days", 0)) > 30
        if order.get("license_revealed"):
            return True
        if order.get("customized") and order.get("damage_reported"):
            return True
        return False

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order)
        is_denial_esc = self._is_denial_or_escalation(order)

        triage = self._triage(ticket_id, thread, snapshot, is_denial_esc)
        retrieved = self._retrieve(order)
        draft = self._compose(ticket_id, thread, snapshot, triage, retrieved)
        final = self._audit(snapshot, draft, is_denial_esc)
        parsed = parse_resolution(final)
        return TicketResolution(parsed.action, parsed.kb_article)

    def _triage(self, ticket_id: str, thread: str, snapshot: str, is_denial_esc: bool) -> str:
        if is_denial_esc:
            return "Triage: policy restriction or escalation review required."
        messages = [
            ChatMessage(
                "system",
                "Triage: classify issue.",
            ),
            ChatMessage(
                "user",
                f"THREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("triage", messages)

    def _retrieve(self, order: dict) -> str:
        ptype = order.get("product_type")
        if ptype == "subscription":
            aid = "subscription-cooling-off"
        elif order.get("license_revealed"):
            aid = "digital-license-no-refund"
        elif order.get("customized") and order.get("damage_reported"):
            aid = "custom-items-no-replace"
        elif order.get("damage_reported"):
            aid = "damage-replacement-7-days"
        else:
            aid = "returns-within-30-days"
        content = load_article(aid).strip().split("\n\n")[0]
        return f"Article: {aid}\n{content}"

    def _compose(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: str,
        retrieved: str,
    ) -> str:
        messages = [
            ChatMessage(
                "system",
                "Compose: output JSON with action, kb_article, customer_reply.",
            ),
            ChatMessage(
                "user",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT\nPOLICY:\n{retrieved}",
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(
        self,
        snapshot: str,
        draft: str,
        is_denial_esc: bool,
    ) -> str:
        if not is_denial_esc:
            return draft
        messages = [
            ChatMessage(
                "system",
                "Audit: verify policy compliance and output JSON with action, kb_article, customer_reply.",
            ),
            ChatMessage(
                "user",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT\nDRAFT:\n{draft}",
            ),
        ]
        return self.llm.chat("audit", messages)
