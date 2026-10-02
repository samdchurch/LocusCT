import sys
from pathlib import Path

# tests/ has no __init__.py, so pytest's default import mode would otherwise
# insert tests/ itself (not the repo root) onto sys.path -- explicit here so
# `from models...`/`from data...` resolve regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
