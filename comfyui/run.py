#!/usr/bin/env python3
"""ComfyUI ワークフロー実行ラッパースクリプト。

例:
    python comfyui/run.py comfyui/workflows/my_workflow.json
    python comfyui/run.py --all --random-seed
"""

from __future__ import annotations

import sys
from pathlib import Path

# リポジトリルート等どのディレクトリから実行しても runner パッケージを解決できるようにする
_COMFYUI_DIR = Path(__file__).resolve().parent
if str(_COMFYUI_DIR) not in sys.path:
    sys.path.insert(0, str(_COMFYUI_DIR))

from runner.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
