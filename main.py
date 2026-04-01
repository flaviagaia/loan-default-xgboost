from __future__ import annotations

import json
from pathlib import Path

from src.modeling import run_project


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    report = run_project(base_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

