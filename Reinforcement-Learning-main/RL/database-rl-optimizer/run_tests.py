#!/usr/bin/env python
"""Run the complete project regression suite."""

import subprocess
import sys


def main():
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
