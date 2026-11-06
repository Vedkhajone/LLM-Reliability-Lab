"""Make scripts runnable from a fresh clone without `pip install -e .`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliability_lab.utils.config import load_env  # noqa: E402

load_env()
