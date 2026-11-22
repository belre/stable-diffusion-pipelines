"""ComfyUI REST / WebSocket API クライアントとワークフロー実行ランナー。"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .workflow import (
    apply_overrides,
    apply_sdxl_preset,
    collect_input_image_nodes,
    load_preset,
    load_workflow,
    validate_api_workflow,
)


class ComfyUIError(RuntimeError):
    """ComfyUI API 通信・実行時のエラー。"""


@dataclass
class RunResult:
    """単一ワークフロー実行の結果。"""

    prompt_id: str
    workflow_path: Path | None
    output_files: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    history: dict[str, Any] = field(default_factory=dict)


def _build_multipart_body(
    fields: Mapping[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    """stdlib のみで multipart/form-data のリクエストボディを生成する。"""
    boundary = f"----ComfyUIRunner{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for key, val in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{val}\r\n".encode("utf-8")
        )

    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    header_content_type = f"multipart/form-data; boundary={boundary}"
    return body, header_content_type


class ComfyUIClient:
    """ComfyUI サーバーと通信してワークフローを実行・画像取得するクライアント。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout

    def _http_request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=data,
            headers=dict(headers) if headers else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.request_timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise ComfyUIError(
                f"ComfyUI HTTP エラー ({exc.code}) {method} {url}: {body_text or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ComfyUIError(
                f"ComfyUI サーバーに接続できません ({url}): {exc.reason}"
            ) from exc

    def _http_get_json(self, path: str, *, timeout: float | None = None) -> Any:
        raw = self._http_request("GET", path, timeout=timeout)
        return json.loads(raw.decode("utf-8"))

    def _http_post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8")
        raw = self._http_request(
            "POST",
            path,
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        return json.loads(raw.decode("utf-8"))

    def is_alive(self) -> bool:
        """ComfyUI サーバーが応答可能か確認する。"""
        try:
            self._http_get_json("/system_stats", timeout=5.0)
            return True
        except ComfyUIError:
            return False

    def wait_until_ready(self, *, timeout: float = 120.0, poll_interval: float = 2.0) -> None:
        """ComfyUI サーバーが応答するまで待機する。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_alive():
                return
            time.sleep(poll_interval)
        raise TimeoutError(
            f"ComfyUI サーバー ({self.base_url}) が {timeout:.0f} 秒以内に起動しませんでした。"
        )

    def upload_image(
        self,
        image_path: Path | str,
        *,
        subfolder: str = "",
        image_type: str = "input",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """ローカル画像を ComfyUI の input ディレクトリへアップロードする。"""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"アップロード対象画像が見つかりません: {path}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        fields: dict[str, str] = {
            "type": image_type,
            "overwrite": "true" if overwrite else "false",
        }
        if subfolder:
            fields["subfolder"] = subfolder

        body, header_ct = _build_multipart_body(
            fields=fields,
            file_field="image",
            filename=path.name,
            file_bytes=path.read_bytes(),
            content_type=content_type,
        )
        raw = self._http_request(
            "POST",
            "/upload/image",
            data=body,
            headers={"Content-Type": header_ct},
        )
        return json.loads(raw.decode("utf-8"))

    def queue_prompt(
        self,
        workflow: Mapping[str, Mapping[str, Any]],
        *,
        client_id: str | None = None,
    ) -> str:
        """API 形式ワークフローを ComfyUI のキューに投入し `prompt_id` を返す。"""
        payload: dict[str, Any] = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id

        response = self._http_post_json("/prompt", payload)
        node_errors = response.get("node_errors")
        if node_errors:
            raise ComfyUIError(
                f"ワークフローのバリデーションエラーが返されました: {json.dumps(node_errors, ensure_ascii=False)}"
            )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"/prompt の応答に prompt_id が含まれていません: {response}")
        return str(prompt_id)

    def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        """指定した `prompt_id` の実行履歴を取得する。未完了時は None を返す。"""
        data = self._http_get_json(f"/history/{urllib.parse.quote(prompt_id)}")
        if isinstance(data, dict) and prompt_id in data:
            return data[prompt_id]
        return None

    def _ws_url(self, client_id: str) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{ws_scheme}://{parsed.netloc}/ws?clientId={urllib.parse.quote(client_id)}"

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        client_id: str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
        use_websocket: bool = True,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> dict[str, Any]:
        """ジョブの完了を待機し、`/history/{prompt_id}` のエントリを返す。"""
        deadline = time.monotonic() + timeout

        if use_websocket and client_id:
            try:
                import websocket  # type: ignore[import-not-found]
            except ImportError:
                websocket = None

            if websocket is not None:
                ws = None
                try:
                    ws = websocket.create_connection(self._ws_url(client_id), timeout=5.0)
                    while time.monotonic() < deadline:
                        try:
                            raw_msg = ws.recv()
                        except websocket.WebSocketTimeoutException:
                            history = self.get_history(prompt_id)
                            if history is not None:
                                return history
                            continue

                        if not isinstance(raw_msg, str):
                            continue
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type")
                        msg_data = msg.get("data", {})

                        if msg_type == "progress" and progress_callback is not None:
                            progress_callback(
                                int(msg_data.get("value", 0)),
                                int(msg_data.get("max", 0)),
                                msg_data.get("node"),
                            )
                        elif msg_type == "execution_error" and msg_data.get("prompt_id") == prompt_id:
                            raise ComfyUIError(
                                f"ComfyUI 実行エラー (node={msg_data.get('node_id')} / "
                                f"{msg_data.get('node_type')}): {msg_data.get('exception_message')}"
                            )
                        elif (
                            msg_type == "executing"
                            and msg_data.get("prompt_id") == prompt_id
                            and msg_data.get("node") is None
                        ):
                            break
                except ComfyUIError:
                    raise
                except Exception:
                    # WebSocket 接続不可時は HTTP ポーリングへフォールバック
                    pass
                finally:
                    if ws is not None:
                        try:
                            ws.close()
                        except Exception:
                            pass

        while time.monotonic() < deadline:
            history = self.get_history(prompt_id)
            if history is not None:
                status = history.get("status", {})
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise ComfyUIError(
                        f"ComfyUI 実行がエラー終了しました (prompt_id={prompt_id}): {status}"
                    )
                return history
            time.sleep(poll_interval)

        raise TimeoutError(
            f"ワークフロー実行 (prompt_id={prompt_id}) が {timeout:.0f} 秒以内に完了しませんでした。"
        )

    def download_file(
        self,
        filename: str,
        *,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """ComfyUI の `/view` エンドポイントから生成ファイルのバイト列を取得する。"""
        params = urllib.parse.urlencode(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            }
        )
        return self._http_request("GET", f"/view?{params}")

    def download_outputs(
        self,
        history_entry: Mapping[str, Any],
        output_dir: Path | str,
        *,
        prefix: str = "",
    ) -> list[Path]:
        """履歴エントリに含まれる全出力ファイル（画像・GIF 等）を `output_dir` へ保存する。"""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        outputs = history_entry.get("outputs", {})
        if not isinstance(outputs, Mapping):
            return saved_paths

        for node_output in outputs.values():
            if not isinstance(node_output, Mapping):
                continue
            for media_key in ("images", "gifs", "videos"):
                items = node_output.get(media_key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, Mapping) or "filename" not in item:
                        continue
                    filename = str(item["filename"])
                    subfolder = str(item.get("subfolder", ""))
                    folder_type = str(item.get("type", "output"))
                    # temp プレビューのみの場合は output のみ保存する運用も考えられるが、
                    # SaveImage は type="output" になる。PreviewImage のみの場合はスキップしない。
                    raw_bytes = self.download_file(
                        filename,
                        subfolder=subfolder,
                        folder_type=folder_type,
                    )
                    target_name = f"{prefix}{filename}" if prefix else filename
                    target_path = _unique_path(out_dir / target_name)
                    target_path.write_bytes(raw_bytes)
                    saved_paths.append(target_path)

        return saved_paths

    def _upload_referenced_inputs(
        self,
        workflow: dict[str, dict[str, Any]],
        input_dir: Path | str | None,
    ) -> None:
        """LoadImage 系ノードが参照するローカル画像を自動アップロードし、登録名に更新する。"""
        base_input_dir = Path(input_dir) if input_dir else None
        for node_id, image_val in collect_input_image_nodes(workflow):
            candidate = Path(image_val)
            resolved: Path | None = None
            if candidate.is_file():
                resolved = candidate
            elif base_input_dir is not None and (base_input_dir / image_val).is_file():
                resolved = base_input_dir / image_val

            if resolved is not None:
                uploaded = self.upload_image(resolved, overwrite=True)
                uploaded_name = uploaded.get("name", resolved.name)
                uploaded_sub = uploaded.get("subfolder", "")
                workflow[node_id]["inputs"]["image"] = (
                    f"{uploaded_sub}/{uploaded_name}" if uploaded_sub else uploaded_name
                )

    def run_workflow(
        self,
        workflow_or_path: Path | str | Mapping[str, Mapping[str, Any]],
        output_dir: Path | str,
        *,
        preset: Path | str | Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        randomize_seed: bool = False,
        seed: int | None = None,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        input_dir: Path | str | None = None,
        timeout: float = 600.0,
        use_websocket: bool = True,
        prefix: str = "",
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> RunResult:
        """ワークフローを読み込み・プリセット適用・上書き・送信・完了待機・出力保存まで一括実行する。"""
        workflow_path: Path | None = None
        if isinstance(workflow_or_path, (str, Path)):
            workflow_path = Path(workflow_or_path)
            raw_workflow = load_workflow(workflow_path)
        else:
            raw_workflow = validate_api_workflow(workflow_or_path)

        base_workflow: Mapping[str, Mapping[str, Any]] = raw_workflow
        if preset is not None:
            preset_dict = load_preset(preset) if isinstance(preset, (str, Path)) else preset
            base_workflow = apply_sdxl_preset(base_workflow, preset_dict)

        prepared = apply_overrides(
            base_workflow,
            overrides=overrides,
            randomize_seed=randomize_seed,
            seed=seed,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
        )

        self._upload_referenced_inputs(prepared, input_dir=input_dir)

        client_id = uuid.uuid4().hex
        start_time = time.monotonic()
        prompt_id = self.queue_prompt(prepared, client_id=client_id)
        history = self.wait_for_completion(
            prompt_id,
            client_id=client_id,
            timeout=timeout,
            use_websocket=use_websocket,
            progress_callback=progress_callback,
        )
        saved_files = self.download_outputs(history, output_dir=output_dir, prefix=prefix)
        elapsed = time.monotonic() - start_time

        return RunResult(
            prompt_id=prompt_id,
            workflow_path=workflow_path,
            output_files=saved_files,
            elapsed_seconds=elapsed,
            history=history,
        )

    def run_directory(
        self,
        workflows_dir: Path | str,
        output_dir: Path | str,
        *,
        pattern: str = "*.json",
        repeat: int = 1,
        **kwargs: Any,
    ) -> list[RunResult]:
        """ディレクトリ内のワークフロー JSON を名前順に一括実行する。"""
        wf_dir = Path(workflows_dir)
        if not wf_dir.is_dir():
            raise FileNotFoundError(f"ワークフローディレクトリが見つかりません: {wf_dir}")

        workflow_files = sorted(p for p in wf_dir.glob(pattern) if p.is_file())
        if not workflow_files:
            return []

        results: list[RunResult] = []
        for wf_file in workflow_files:
            for idx in range(repeat):
                run_kwargs = dict(kwargs)
                if not run_kwargs.get("prefix"):
                    suffix = f"_{idx + 1:02d}" if repeat > 1 else ""
                    run_kwargs["prefix"] = f"{wf_file.stem}{suffix}_"
                results.append(self.run_workflow(wf_file, output_dir, **run_kwargs))
        return results


def _unique_path(path: Path) -> Path:
    """同名ファイルが既に存在する場合に連番サフィックスを付与する。"""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter:03d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
