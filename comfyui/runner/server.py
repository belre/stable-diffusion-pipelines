"""ComfyUI サーバーのバックグラウンド起動・モデルパス連携ヘルパー（主に Google Colab 向け）。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .client import ComfyUIClient

DEFAULT_MODEL_SUBDIRS: tuple[str, ...] = (
    "checkpoints",
    "loras",
    "vae",
    "clip",
    "clip_vision",
    "controlnet",
    "diffusion_models",
    "unet",
    "embeddings",
    "upscale_models",
    "ipadapter",
)


def generate_extra_model_paths_yaml(
    drive_models_dir: Path | str,
    output_path: Path | str,
    *,
    subdirs: Sequence[str] = DEFAULT_MODEL_SUBDIRS,
) -> Path:
    """Google Drive 等のモデルフォルダを参照する `extra_model_paths.yaml` を生成する。

    ComfyUI 起動時に `--extra-model-paths-config <output_path>` を渡すことで、
    大容量モデルをローカルへコピーせずに直接読み込める。
    """
    base_dir = Path(drive_models_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "external_models:",
        f"    base_path: {base_dir.as_posix()}",
    ]
    for sub in subdirs:
        lines.append(f"    {sub}: {sub}/")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def link_drive_models(
    comfyui_dir: Path | str,
    drive_models_dir: Path | str,
    *,
    subdirs: Sequence[str] = DEFAULT_MODEL_SUBDIRS,
) -> dict[str, list[Path]]:
    """外部モデルフォルダ配下のファイルを `ComfyUI/models/<subdir>/` にシンボリックリンクする。"""
    comfy_models = Path(comfyui_dir) / "models"
    src_root = Path(drive_models_dir)
    created: dict[str, list[Path]] = {}

    for sub in subdirs:
        src_dir = src_root / sub
        if not src_dir.is_dir():
            continue
        dst_dir = comfy_models / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        linked_files: list[Path] = []
        for item in sorted(src_dir.iterdir()):
            if item.name.startswith("."):
                continue
            dst_link = dst_dir / item.name
            if dst_link.exists() or dst_link.is_symlink():
                continue
            os.symlink(item, dst_link)
            linked_files.append(dst_link)
        if linked_files:
            created[sub] = linked_files

    return created


class ComfyUIServer:
    """ComfyUI (`main.py`) をバックグラウンドプロセスとして起動・管理するクラス。"""

    def __init__(
        self,
        comfyui_dir: Path | str,
        *,
        host: str = "127.0.0.1",
        port: int = 8188,
        python_executable: str | None = None,
        extra_args: Sequence[str] | None = None,
        extra_model_paths_config: Path | str | None = None,
        log_file: Path | str | None = None,
    ) -> None:
        self.comfyui_dir = Path(comfyui_dir)
        self.host = host
        self.port = port
        self.python_executable = python_executable or sys.executable
        self.extra_args = list(extra_args) if extra_args else []
        self.extra_model_paths_config = (
            Path(extra_model_paths_config) if extra_model_paths_config else None
        )
        self.log_file = Path(log_file) if log_file else self.comfyui_dir / "comfyui_server.log"
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_command(self) -> list[str]:
        """ComfyUI 起動コマンドを構築する。"""
        cmd = [
            self.python_executable,
            "main.py",
            "--listen",
            self.host,
            "--port",
            str(self.port),
            "--disable-auto-launch",
        ]
        if self.extra_model_paths_config is not None:
            cmd.extend(["--extra-model-paths-config", str(self.extra_model_paths_config)])
        cmd.extend(self.extra_args)
        return cmd

    def start(self, *, wait_ready: bool = True, timeout: float = 180.0) -> ComfyUIClient:
        """ComfyUI サーバーを起動し、準備完了まで待機してクライアントを返す。"""
        main_py = self.comfyui_dir / "main.py"
        if not main_py.is_file():
            raise FileNotFoundError(
                f"ComfyUI の main.py が見つかりません: {main_py}"
            )

        client = ComfyUIClient(self.base_url)
        if client.is_alive():
            return client

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_file.open("wb")

        cmd = self.build_command()
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.comfyui_dir),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )

        if wait_ready:
            self._wait_until_ready(client, timeout=timeout)

        return client

    def _wait_until_ready(self, client: ComfyUIClient, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                tail = self.read_log_tail(40)
                raise RuntimeError(
                    f"ComfyUI サーバーが起動途中で終了しました (exit code={self.process.returncode})。\n"
                    f"--- ログ末尾 ({self.log_file}) ---\n{tail}"
                )
            if client.is_alive():
                return
            time.sleep(2.0)

        tail = self.read_log_tail(40)
        raise TimeoutError(
            f"ComfyUI サーバーが {timeout:.0f} 秒以内に応答しませんでした ({self.base_url})。\n"
            f"--- ログ末尾 ({self.log_file}) ---\n{tail}"
        )

    def read_log_tail(self, lines: int = 30) -> str:
        """サーバーログの末尾 N 行を返す。"""
        if not self.log_file.is_file():
            return "(ログファイルなし)"
        text = self.log_file.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def stop(self, timeout: float = 10.0) -> None:
        """起動中の ComfyUI サーバーを停止する。"""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self.process = None

        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> ComfyUIServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
