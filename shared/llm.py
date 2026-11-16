from __future__ import annotations

import json
import re
import textwrap
import uuid
from dataclasses import dataclass

from eval.reference import decide_order
from shared.billing import CACHE_MIN_PREFIX, CostSession
from shared.tokens import count_tokens

_VERBOSE_HINTS = (
    "step by step",
    "full reasoning",
    "detailed plan",
    "double-check",
    "restate the ticket",
    "quote the",
)


@dataclass
class ChatMessage:
    role: str
    content: str


class LlmClient:
    """Production-shaped client. Swap `model=` for a real API; billing stays the same."""

    def __init__(self, session: CostSession, model: str = "gpt-4.1-mini"):
        self.session = session
        self.model = model

    def chat(
        self,
        label: str,
        messages: list[ChatMessage],
        *,
        cached_prefix: str | None = None,
    ) -> str:
        prompt = "\n\n".join(f"{m.role.upper()}:\n{m.content}" for m in messages)
        snapshot = _extract_snapshot(prompt)
        if snapshot is None:
            raise ValueError(
                f"LLM simulator did not find ORDER_SNAPSHOT in call '{label}'. "
                "Include ORDER_SNAPSHOT … END_SNAPSHOT or the simulator cannot run this call."
            )
        outcome = decide_order(snapshot)

        cached_prefix_tokens = 0
        if cached_prefix and count_tokens(cached_prefix) >= CACHE_MIN_PREFIX:
            if prompt.startswith(cached_prefix):
                cached_prefix_tokens = count_tokens(cached_prefix)

        if _wants_verbose(prompt):
            completion = _verbose_json(outcome.action, outcome.kb_article)
        else:
            completion = json.dumps(
                {
                    "action": outcome.action,
                    "kb_article": outcome.kb_article,
                    "customer_reply": _short_reply(outcome.action),
                },
                indent=2,
            )

        self.session.record(label, prompt, completion, cached_prefix_tokens)
        return completion


def _extract_snapshot(prompt: str) -> dict | None:
    m = re.search(
        r"ORDER_SNAPSHOT:\s*(\{.*?\})\s*END_SNAPSHOT",
        prompt,
        re.DOTALL,
    )
    if not m:
        return None
    return json.loads(m.group(1))


def _wants_verbose(prompt: str) -> bool:
    lower = prompt.lower()
    return any(h in lower for h in _VERBOSE_HINTS)


def _short_reply(action: str) -> str:
    return {
        "REFUND": "We approved your refund; you'll see it in 3-5 business days.",
        "REPLACE": "A replacement is on the way.",
        "SEND_ARTICLE": "Here's the policy that applies to your request.",
        "ESCALATE_TIER2": "A specialist will follow up shortly.",
    }.get(action, "Thanks for your patience.")


def _verbose_json(action: str, kb_article: str) -> str:
    body = textwrap.dedent(
        f"""
        Analysis trace {uuid.uuid4()} (internal).

        I reviewed the conversation, compared it to help-center articles, and matched
        order facts to routing rules. Escalation paths were considered.

        {{
          "action": "{action}",
          "kb_article": "{kb_article}",
          "customer_reply": "{_short_reply(action)}",
          "reasoning": "Long internal narrative the product does not need."
        }}
        """
    )
    while count_tokens(body) < 480:
        body += "Additional audit note: verify SKU, delivery age, digital license state, customization.\n"
    return body
