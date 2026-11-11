import os
import sys

# Make "modules" and "ui" importable no matter where the command is run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dashboard import run  # noqa: E402

if __name__ == "__main__":
    run()