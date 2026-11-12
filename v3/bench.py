"""Entry point of the v3 harness. See pcbench/cli.py for the commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
