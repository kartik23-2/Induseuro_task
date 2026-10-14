import sys
from pathlib import Path

# Ensure question_2 and workspace root are in sys.path
project_root = Path(__file__).resolve().parent
workspace_root = project_root.parent

for p in [str(project_root), str(workspace_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
