"""Run a frozen toy experiment in a separate process (so a test can kill it)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from v3_helpers import WordTok, fake_env

from pcbench import runner

sys.exit(runner.run_experiment(Path(sys.argv[1]), tok=WordTok(), env=fake_env()))
