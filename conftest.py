"""pytest configuration for the workspace root.

Adds the workspace root to sys.path so that ``scene_motion_llm`` is
importable without installing the package or setting PYTHONPATH manually.
"""

import sys
from pathlib import Path

# Insert workspace root at the front of sys.path once per session.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
