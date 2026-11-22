"""ComfyUI ワークフロー実行 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .client import ComfyUIClient, ComfyUIError, RunResult
from .workflow import WorkflowFormatError, parse_override_spec

COMFYUI_ROOT = Path(__file__).resolve().parent.parent


def _load_structured_file(path: Path) -> dict[str, Any]:
    """JSON または YAML ファイルを辞書として読み込む。"""
    if not path.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "YAML ファイルを読み込むには PyYAML が必要です: pip install PyYAML"
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"未対応の設定ファイル形式です: {path.suffix} (.json / .yaml / .yml のみ)")

    if not isinstance(data, dict):
        raise ValueError(f"設定ファイルのトップレベルはオブジェクトである必要があります: {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfyui-runner",
        description="ComfyUI API 形式ワークフロー JSON を実行して画像を生成・保存します。",
    )
    parser.add_argument(
        "workflows",
        nargs="*",
        help="実行するワークフロー JSON ファイルまたはディレクトリ。省略時は --all 指定で workflows/ を実行。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="workflows ディレクトリ内の全 .json ファイルを一括実行する。",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="ComfyUI サーバー URL（既定: http://127.0.0.1:8188 または config.yaml の値）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="設定ファイル (.yaml / .json)。",
    )
    parser.add_argument(
        "--workflows-dir",
        default=None,
        help="--all 実行時の探索ディレクトリ（既定: comfyui/workflows）。",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        default=None,
        help="LoadImage 等で参照するローカル入力画像のディレクトリ（既定: comfyui/input）。",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=None,
        help="生成画像の保存先ディレクトリ（既定: comfyui/output）。",
    )
    parser.add_argument(
        "--positive",
        default=None,
        help="Positive プロンプトを上書きする文字列。",
    )
    parser.add_argument(
        "--negative",
        default=None,
        help="Negative プロンプトを上書きする文字列。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="シード値を固定値で上書きする。",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="実行ごとにシード値をランダム化する。",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="SELECTOR=VALUE",
        help=(
            "任意ノードの入力を上書きする（複数指定可）。"
            "例: --set 6.text='1girl' / --set title:KSampler.steps=28 / --set class:KSampler.cfg=7.0"
        ),
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="上書き定義をまとめた JSON / YAML ファイル。",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="SDXL 等のパラメータプリセット（YAML / JSON）。例: comfyui/presets/sdxl_default.yaml",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="各ワークフローの繰り返し実行回数（既定: 1）。",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="保存ファイル名の先頭プレフィックス。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="1 ワークフローあたりのタイムアウト秒数（既定: 600）。",
    )
    parser.add_argument(
        "--no-ws",
        action="store_true",
        help="WebSocket 進捗監視を無効にし、HTTP ポーリングのみで待機する。",
    )
    return parser


def _resolve_config_path(raw: str | None, base_dir: Path, default: Path) -> Path:
    if raw is None:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config: dict[str, Any] = {}
    config_base = COMFYUI_ROOT
    if args.config:
        cfg_path = Path(args.config)
        config = _load_structured_file(cfg_path)
        config_base = cfg_path.resolve().parent
    elif (COMFYUI_ROOT / "config.yaml").is_file():
        config = _load_structured_file(COMFYUI_ROOT / "config.yaml")

    server_cfg = config.get("server", {}) if isinstance(config.get("server"), dict) else {}
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    defaults_cfg = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}

    base_url = args.url or server_cfg.get("url") or "http://127.0.0.1:8188"
    timeout = float(args.timeout if args.timeout is not None else server_cfg.get("timeout", 600.0))
    use_ws = (not args.no_ws) and bool(server_cfg.get("use_websocket", True))

    workflows_dir = (
        Path(args.workflows_dir)
        if args.workflows_dir
        else _resolve_config_path(paths_cfg.get("workflows_dir"), config_base, COMFYUI_ROOT / "workflows")
    )
    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else _resolve_config_path(paths_cfg.get("input_dir"), config_base, COMFYUI_ROOT / "input")
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _resolve_config_path(paths_cfg.get("output_dir"), config_base, COMFYUI_ROOT / "output")
    )

    # 上書き辞書のマージ（config.defaults.overrides -> --overrides -> --set）
    overrides: dict[str, Any] = {}
    if isinstance(defaults_cfg.get("overrides"), dict):
        overrides.update(defaults_cfg["overrides"])
    if args.overrides:
        overrides.update(_load_structured_file(Path(args.overrides)))
    for spec in args.set_overrides:
        k, v = parse_override_spec(spec)
        overrides[k] = v

    randomize_seed = bool(args.random_seed or defaults_cfg.get("randomize_seed", False))
    if args.repeat < 1:
        parser.error("--repeat は 1 以上を指定してください。")

    # 対象ワークフローの解決
    targets: list[Path] = []
    if args.all:
        if workflows_dir.is_dir():
            targets.extend(sorted(p for p in workflows_dir.glob("*.json") if p.is_file()))
    for item in args.workflows:
        p = Path(item)
        if p.is_dir():
            targets.extend(sorted(f for f in p.glob("*.json") if f.is_file()))
        else:
            targets.append(p)

    if not targets:
        print(
            f"実行対象のワークフロー JSON が見つかりません。"
            f"ファイルを指定するか、{workflows_dir} に JSON を配置して --all を付けて実行してください。",
            file=sys.stderr,
        )
        return 1

    client = ComfyUIClient(base_url=base_url)
    if not client.is_alive():
        print(
            f"[ERROR] ComfyUI サーバーに接続できません: {base_url}\n"
            "サーバーが起動しているか確認するか、--url で接続先を指定してください。",
            file=sys.stderr,
        )
        return 1

    all_results: list[RunResult] = []
    try:
        for wf_path in targets:
            for idx in range(args.repeat):
                run_prefix = args.prefix
                if not run_prefix and (len(targets) > 1 or args.repeat > 1):
                    suffix = f"_{idx + 1:02d}" if args.repeat > 1 else ""
                    run_prefix = f"{wf_path.stem}{suffix}_"

                print(f"[RUN] {wf_path} ({idx + 1}/{args.repeat}) -> {base_url}")
                result = client.run_workflow(
                    wf_path,
                    output_dir=output_dir,
                    preset=args.preset,
                    overrides=overrides or None,
                    randomize_seed=randomize_seed,
                    seed=args.seed,
                    positive_prompt=args.positive,
                    negative_prompt=args.negative,
                    input_dir=input_dir if input_dir.is_dir() else None,
                    timeout=timeout,
                    use_websocket=use_ws,
                    prefix=run_prefix,
                )
                all_results.append(result)
                print(
                    f"[DONE] prompt_id={result.prompt_id} "
                    f"({result.elapsed_seconds:.1f}s) -> {len(result.output_files)} file(s)"
                )
                for out_file in result.output_files:
                    print(f"       saved: {out_file}")
    except (WorkflowFormatError, ComfyUIError, FileNotFoundError, KeyError, ValueError, TimeoutError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    total_files = sum(len(r.output_files) for r in all_results)
    print(f"完了: {len(all_results)} ジョブ実行 / {total_files} ファイルを {output_dir} に保存しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
