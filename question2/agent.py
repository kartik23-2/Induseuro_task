from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import expanded_help_center_blob, title_index
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
        self.help_center = expanded_help_center_blob()

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, indent=2)

        triage = self._triage(ticket_id, thread, snapshot)
        retrieved = self._retrieve(triage, snapshot)
        draft = self._compose(ticket_id, thread, snapshot, triage, retrieved)
        final = self._audit(ticket_id, thread, snapshot, triage, retrieved, draft)
        parsed = parse_resolution(final)
        return TicketResolution(parsed.action, parsed.kb_article)

    def _triage(self, ticket_id: str, thread: str, snapshot: str) -> str:
        messages = [
            ChatMessage(
                "system",
                "\n".join(
                    [
                        f"trace-{uuid.uuid4()}",
                        "Triage node: restate the ticket step by step and list candidate kb articles.",
                        f"ARTICLE TITLES:\n{title_index(500)}",
                        self.help_center[:120_000],
                    ]
                ),
            ),
            ChatMessage(
                "user",
                f"TICKET {ticket_id}\nTHREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("triage", messages)

    def _retrieve(self, triage: str, snapshot: str) -> str:
        messages = [
            ChatMessage(
                "system",
                "Retrieve node: print full text of the eight most relevant help articles.",
            ),
            ChatMessage(
                "user",
                f"TRIAGE:\n{triage}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("retrieve", messages)

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
                "Compose node: full reasoning, then JSON with action, kb_article, customer_reply.",
            ),
            ChatMessage(
                "user",
                "\n\n".join(
                    [
                        f"TICKET {ticket_id}",
                        f"THREAD:\n{thread}",
                        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                        f"TRIAGE:\n{triage}",
                        f"ARTICLES:\n{retrieved}",
                        f"TRIAGE AGAIN:\n{triage}",
                    ]
                ),
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: str,
        retrieved: str,
        draft: str,
    ) -> str:
        messages = [
            ChatMessage(
                "system",
                "Audit node: double-check policy, restate ticket, rewrite JSON output.",
            ),
            ChatMessage(
                "user",
                "\n\n".join(
                    [
                        self.help_center[:80_000],
                        f"TICKET {ticket_id}",
                        f"THREAD:\n{thread}",
                        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                        f"DRAFT:\n{draft}",
                        f"TRIAGE:\n{triage}",
                        f"ARTICLES:\n{retrieved}",
                    ]
                ),
            ),
        ]
        return self.llm.chat("audit", messages)
