from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from question2.agent import SupportPipeline  # noqa: E402

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def main() -> None:
    pipe = SupportPipeline()
    rows = []
    for tid in PUBLIC:
        res = pipe.handle_ticket(tid)
        rows.append({"ticket_id": tid, "action": res.action, "kb_article": res.kb_article})
    print(json.dumps({"resolutions": rows, "total_points": pipe.session.total_points()}, indent=2))


if __name__ == "__main__":
    main()
