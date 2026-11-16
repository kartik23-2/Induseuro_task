from __future__ import annotations

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def test_baseline_resolves_public_tickets():
    copilot = SupportCopilot()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = copilot.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_holdout_subscription():
    copilot = SupportCopilot()
    got = copilot.handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert got.action == exp.action
    assert got.kb_article == exp.kb_article


def test_baseline_cost_is_high():
    copilot = SupportCopilot()
    for tid in PUBLIC:
        copilot.handle_ticket(tid)
    assert copilot.session.total_points() > 15_000
