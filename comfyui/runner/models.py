"""SDXL 等のモデル重みファイル（Checkpoint / LoRA / VAE 等）のダウンロード・存在確認ユーティリティ。

ハイブリッド運用を想定:
- 公開ベースモデル（SDXL Checkpoint 等）: Hugging Face / Civitai / 直接 URL からローカルへ自動 DL
- 自作 LoRA・プライベートモデル: Google Drive 側 (`extra_model_paths.yaml`) から参照
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_NODE_FIELD_MAP: dict[str, tuple[str, str]] = {
    "CheckpointLoaderSimple": ("checkpoints", "ckpt_name"),
    "CheckpointLoader": ("checkpoints", "ckpt_name"),
    "LoraLoader": ("loras", "lora_name"),
    "LoraLoaderModelOnly": ("loras", "lora_name"),
    "VAELoader": ("vae", "vae_name"),
    "ControlNetLoader": ("controlnet", "control_net_name"),
    "UNETLoader": ("diffusion_models", "unet_name"),
    "UpscaleModelLoader": ("upscale_models", "model_name"),
}


@dataclass
class ModelSpec:
    """ダウンロード対象のモデル定義。"""

    subdir: str
    filename: str
    url: str | None = None
    hf_repo: str | None = None
    hf_file: str | None = None
    hf_revision: str = "main"

    def resolve_url(self) -> str:
        """ダウンロード用 URL を解決する。"""
        if self.url:
            return self.url
        if self.hf_repo and self.hf_file:
            repo = self.hf_repo.strip("/")
            file_path = self.hf_file.lstrip("/")
            rev = urllib.parse.quote(self.hf_revision, safe="")
            return f"https://huggingface.co/{repo}/resolve/{rev}/{file_path}"
        raise ValueError(
            f"モデル '{self.filename}' には 'url' または 'hf_repo' + 'hf_file' の指定が必要です。"
        )

    @property
    def is_huggingface(self) -> bool:
        if self.hf_repo and self.hf_file:
            return True
        return bool(self.url and "huggingface.co" in urllib.parse.urlparse(self.url).netloc)

    @property
    def is_civitai(self) -> bool:
        return bool(self.url and "civitai.com" in urllib.parse.urlparse(self.url).netloc)


def load_model_manifest(path: Path | str) -> list[ModelSpec]:
    """YAML / JSON のモデルマニフェストを読み込み `ModelSpec` のリストを返す。"""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"モデルマニフェストが見つかりません: {manifest_path}")

    text = manifest_path.read_text(encoding="utf-8")
    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("YAML を読み込むには PyYAML が必要です: pip install PyYAML") from exc
        raw = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"未対応のファイル形式です: {manifest_path.suffix}")

    entries = raw.get("models", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("モデルマニフェストは 'models' リストを含む必要があります。")

    specs: list[ModelSpec] = []
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        if item.get("enabled", True) is False:
            continue
        specs.append(
            ModelSpec(
                subdir=str(item["subdir"]),
                filename=str(item["filename"]),
                url=item.get("url"),
                hf_repo=item.get("hf_repo"),
                hf_file=item.get("hf_file"),
                hf_revision=str(item.get("hf_revision", "main")),
            )
        )
    return specs


def _append_civitai_token(url: str, token: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if "token" not in query:
        query["token"] = [token]
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _download_file(
    url: str,
    dest: Path,
    *,
    headers: Mapping[str, str] | None = None,
    prefer_aria2: bool = True,
) -> Path:
    """ファイルを一時 `.part` にダウンロードし、完了後に原子的にリネームする。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".part")

    if prefer_aria2 and shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "--console-log-level=warn",
            "--allow-overwrite=true",
            "-c",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "-d",
            str(dest.parent),
            "-o",
            tmp_path.name,
        ]
        for k, v in (headers or {}).items():
            cmd.extend(["--header", f"{k}: {v}"])
        cmd.append(url)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            if result.returncode == 24:
                raise RuntimeError(
                    f"ダウンロード認証エラー (aria2c exit 24 / HTTP 401): {dest.name}\n"
                    "Civitai の場合は Colab Secrets に `CIVITAI_API_KEY` を登録し、"
                    "「ノートブックからのアクセス」を ON にしてください。\n"
                    "Hugging Face の Gated モデルの場合は `HF_TOKEN` を設定してください。"
                )
            if result.returncode == 22:
                raise RuntimeError(
                    f"ダウンロード HTTP エラー (aria2c exit 22 / HTTP 403・404 等): {dest.name}\n"
                    "URL（モデルバージョン ID）が正しいか、または認証トークンが必要か確認してください。"
                )
            raise RuntimeError(
                f"aria2c ダウンロードに失敗しました (exit code={result.returncode}): {dest.name}"
            )
        tmp_path.replace(dest)
        return dest

    req = urllib.request.Request(url, headers=dict(headers) if headers else {})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp_path.replace(dest)
    return dest


def download_models(
    specs: Sequence[ModelSpec],
    comfyui_dir: Path | str,
    *,
    hf_token: str | None = None,
    civitai_token: str | None = None,
    drive_models_dir: Path | str | None = None,
    prefer_aria2: bool = True,
    skip_existing: bool = True,
) -> list[Path]:
    """定義されたモデルを `ComfyUI/models/<subdir>/` へダウンロードする。

    `drive_models_dir` 側またはローカル側に既に同名ファイルがある場合はダウンロードをスキップする。
    """
    comfy_models = Path(comfyui_dir) / "models"
    drive_models = Path(drive_models_dir) if drive_models_dir else None
    results: list[Path] = []

    for spec in specs:
        local_dest = comfy_models / spec.subdir / spec.filename
        if skip_existing and local_dest.is_file():
            print(f"[SKIP] ローカルに存在します: {local_dest}")
            results.append(local_dest)
            continue

        if skip_existing and drive_models is not None:
            drive_candidate = drive_models / spec.subdir / spec.filename
            if drive_candidate.is_file():
                print(f"[SKIP] Drive に存在します: {drive_candidate}")
                results.append(drive_candidate)
                continue

        url = spec.resolve_url()
        headers: dict[str, str] = {}
        if spec.is_huggingface and hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        elif spec.is_civitai:
            if civitai_token:
                url = _append_civitai_token(url, civitai_token)
            elif "token=" not in url:
                print(
                    f"[WARN] {spec.filename}: Civitai のダウンロードには CIVITAI_API_KEY が必要な場合があります。"
                    "Colab Secrets の登録と「ノートブックからのアクセス」ON を確認してください。"
                )

        print(f"[DOWNLOAD] {spec.subdir}/{spec.filename}")
        downloaded = _download_file(url, local_dest, headers=headers, prefer_aria2=prefer_aria2)
        results.append(downloaded)

    return results


def check_workflow_models(
    workflow: Mapping[str, Mapping[str, Any]],
    comfyui_dir: Path | str,
    *,
    drive_models_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """ワークフロー JSON が参照するモデルファイルの有無を事前チェックする。

    Returns
    -------
    dict[str, list[str]]
        `{"found": [...], "missing": [...]}` 形式のチェック結果。
    """
    search_roots = [Path(comfyui_dir) / "models"]
    if drive_models_dir:
        search_roots.append(Path(drive_models_dir))

    found: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    for node in workflow.values():
        class_type = node.get("class_type")
        if class_type not in MODEL_NODE_FIELD_MAP:
            continue
        subdir, field_name = MODEL_NODE_FIELD_MAP[class_type]
        val = node.get("inputs", {}).get(field_name)
        if not isinstance(val, str) or not val.strip():
            continue

        rel_key = f"{subdir}/{val}"
        if rel_key in seen:
            continue
        seen.add(rel_key)

        exists = any((root / subdir / val).is_file() for root in search_roots)
        if exists:
            found.append(rel_key)
        else:
            missing.append(rel_key)

    return {"found": found, "missing": missing}
