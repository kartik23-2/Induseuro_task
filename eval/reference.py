from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedOutcome:
    action: str
    kb_article: str


def decide_order(order: dict) -> ExpectedOutcome:
    """Ground truth for grading (used by tests and the local LLM simulator)."""
    ptype = order.get("product_type")
    if ptype == "subscription":
        days = int(order.get("subscription_age_days", 0))
        if days <= 30:
            return ExpectedOutcome("REFUND", "subscription-cooling-off")
        return ExpectedOutcome("SEND_ARTICLE", "subscription-cooling-off")

    if order.get("license_revealed"):
        return ExpectedOutcome("SEND_ARTICLE", "digital-license-no-refund")

    if order.get("customized") and order.get("damage_reported"):
        return ExpectedOutcome("ESCALATE_TIER2", "custom-items-no-replace")

    if order.get("damage_reported") and not order.get("customized"):
        if int(order.get("delivered_days_ago", 99)) <= 7:
            return ExpectedOutcome("REPLACE", "damage-replacement-7-days")

    if ptype == "physical" and not order.get("opened"):
        if int(order.get("delivered_days_ago", 99)) <= 30:
            return ExpectedOutcome("REFUND", "returns-within-30-days")

    return ExpectedOutcome("ESCALATE_TIER2", "returns-within-30-days")


TICKET_EXPECTATIONS: dict[str, ExpectedOutcome] = {
    "T-1001": ExpectedOutcome("REFUND", "returns-within-30-days"),
    "T-1002": ExpectedOutcome("SEND_ARTICLE", "digital-license-no-refund"),
    "T-1003": ExpectedOutcome("ESCALATE_TIER2", "custom-items-no-replace"),
    "T-1004": ExpectedOutcome("REPLACE", "damage-replacement-7-days"),
    "T-1091": ExpectedOutcome("SEND_ARTICLE", "subscription-cooling-off"),
}
