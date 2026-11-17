from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import all_articles_blob
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


_FEW_SHOT = """
Example ticket resolutions (format to copy):

1. Unopened headphones, day 10 → {"action":"REFUND","kb_article":"returns-within-30-days"}
2. Ebook, license shown → {"action":"SEND_ARTICLE","kb_article":"digital-license-no-refund"}
3. Cracked custom mug → {"action":"ESCALATE_TIER2","kb_article":"custom-items-no-replace"}
4. Cracked stock mug, day 2 → {"action":"REPLACE","kb_article":"damage-replacement-7-days"}
"""


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (baseline).

    Given a support ticket + order snapshot, choose an action, cite the help-center
    article, and draft the customer reply.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, indent=2)

        system = "\n\n".join(
            [
                # Baseline dynamic prefix (commented out):
                # f"Run id {uuid.uuid4()}",
                "You are ShopAssist, the Tier-1 customer support copilot.",
                _FEW_SHOT,
                "HELP CENTER:",
                all_articles_blob(),
                # Baseline verbose reasoning instructions (commented out):
                # (
                #     "Instructions: Think step by step. Write full reasoning, quote any article "
                #     "you rely on, then output JSON with action, kb_article, customer_reply."
                # ),
                (
                    "Choose the correct action and help-center article. "
                    "Output JSON with action, kb_article, customer_reply."
                ),
            ]
        )
        user = "\n\n".join(
            [
                f"TICKET_ID: {ticket_id}",
                f"THREAD:\n{thread}",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ]
        )
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
