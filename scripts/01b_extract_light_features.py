from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("01_extract_features.py")
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
