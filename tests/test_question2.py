from __future__ import annotations

from shared.kb import expanded_help_center_blob
from shared.tokens import count_tokens
from eval.reference import TICKET_EXPECTATIONS
from question2.agent import SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def test_help_center_exceeds_context():
    assert count_tokens(expanded_help_center_blob()) > 200_000


def test_pipeline_resolves_public_tickets():
    pipe = SupportPipeline()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = pipe.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_baseline_cost_is_high():
    pipe = SupportPipeline()
    pipe.handle_ticket(PUBLIC[0])
    assert pipe.session.total_points() > 70_000
