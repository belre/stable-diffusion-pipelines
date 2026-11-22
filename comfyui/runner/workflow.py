"""ComfyUI API 形式ワークフロー JSON の読み込み・検証・パラメータ上書きユーティリティ。"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping


class WorkflowFormatError(ValueError):
    """ワークフロー JSON が ComfyUI API 形式でない場合に送出される例外。"""


SEED_FIELD_NAMES = ("seed", "noise_seed")
SAMPLER_CLASS_TYPES = {
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
    "RandomNoise",
}
LOAD_IMAGE_CLASS_TYPES = {
    "LoadImage",
    "LoadImageMask",
}


def validate_api_workflow(data: Any) -> dict[str, dict[str, Any]]:
    """ロード済み JSON が ComfyUI の API 形式であることを検証する。

    Parameters
    ----------
    data : Any
        JSON からデコードしたオブジェクト。

    Returns
    -------
    dict[str, dict[str, Any]]
        検証済みの API 形式ワークフロー辞書。

    Raises
    ------
    WorkflowFormatError
        UI 形式（通常の Save）など API 形式以外だった場合。
    """
    if not isinstance(data, dict) or not data:
        raise WorkflowFormatError(
            "ワークフロー JSON は空でない JSON オブジェクトである必要があります。"
        )

    # ComfyUI の通常保存（UI 形式）を検知して分かりやすいエラーを出す
    if "nodes" in data and isinstance(data["nodes"], list):
        raise WorkflowFormatError(
            "通常形式（UI 保存）のワークフロー JSON が指定されました。\n"
            "ComfyUI のメニューから API 形式（Workflow -> Export (API) または "
            "Dev mode 有効化後の Save (API Format)）で書き出した JSON を使用してください。"
        )

    for node_id, node in data.items():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            raise WorkflowFormatError(
                f"ノード '{node_id}' が ComfyUI API 形式ではありません "
                "('class_type' と 'inputs' が必要です)。"
            )
        if not isinstance(node["inputs"], dict):
            raise WorkflowFormatError(
                f"ノード '{node_id}' の 'inputs' はオブジェクトである必要があります。"
            )

    return data


def load_workflow(path: Path | str) -> dict[str, dict[str, Any]]:
    """ファイルから ComfyUI API 形式のワークフロー JSON を読み込む。"""
    workflow_path = Path(path)
    if not workflow_path.is_file():
        raise FileNotFoundError(f"ワークフローファイルが見つかりません: {workflow_path}")

    with workflow_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        return validate_api_workflow(data)
    except WorkflowFormatError as exc:
        raise WorkflowFormatError(f"[{workflow_path}] {exc}") from exc


def find_nodes(
    workflow: Mapping[str, Mapping[str, Any]],
    *,
    node_id: str | None = None,
    title: str | None = None,
    class_type: str | None = None,
) -> list[str]:
    """条件に一致するノード ID のリストを返す。

    Parameters
    ----------
    workflow : Mapping[str, Mapping[str, Any]]
        API 形式ワークフロー。
    node_id : str | None
        完全一致するノード ID。
    title : str | None
        `_meta.title`（大文字小文字を区別しない完全一致）。
    class_type : str | None
        `class_type` の完全一致。
    """
    matched: list[str] = []
    title_lower = title.lower() if title is not None else None

    for nid, node in workflow.items():
        if node_id is not None and nid != node_id:
            continue
        if class_type is not None and node.get("class_type") != class_type:
            continue
        if title_lower is not None:
            meta = node.get("_meta")
            node_title = meta.get("title", "") if isinstance(meta, dict) else ""
            if str(node_title).lower() != title_lower:
                continue
        matched.append(nid)

    return matched


def _resolve_prompt_node_ids(
    workflow: Mapping[str, Mapping[str, Any]],
    kind: str,
) -> list[str]:
    """positive / negative プロンプト対象のノード ID を推定する。

    優先順位:
    1. KSampler 系ノードの `positive` / `negative` リンク先ノード
    2. `_meta.title` に `positive` / `negative` を含むテキストノード
    """
    assert kind in ("positive", "negative")
    linked_ids: list[str] = []

    for node in workflow.values():
        if node.get("class_type") in SAMPLER_CLASS_TYPES:
            link = node.get("inputs", {}).get(kind)
            if isinstance(link, list) and link:
                target_id = str(link[0])
                target = workflow.get(target_id)
                if target and _has_text_input(target) and target_id not in linked_ids:
                    linked_ids.append(target_id)

    if linked_ids:
        return linked_ids

    title_ids: list[str] = []
    for nid, node in workflow.items():
        meta = node.get("_meta")
        node_title = str(meta.get("title", "")).lower() if isinstance(meta, dict) else ""
        if kind in node_title and _has_text_input(node):
            title_ids.append(nid)

    return title_ids


def _has_text_input(node: Mapping[str, Any]) -> bool:
    inputs = node.get("inputs", {})
    if isinstance(inputs.get("text"), str):
        return True
    # SDXL 用 CLIPTextEncodeSDXL は text_g / text_l を持つ
    return isinstance(inputs.get("text_g"), str) or isinstance(inputs.get("text_l"), str)


def _set_prompt_text(node: dict[str, Any], prompt_text: str) -> None:
    """通常 CLIPTextEncode (`text`) および CLIPTextEncodeSDXL (`text_g`/`text_l`) の両方に書き込む。"""
    inputs = node.setdefault("inputs", {})
    if isinstance(inputs.get("text"), str):
        inputs["text"] = prompt_text
    if isinstance(inputs.get("text_g"), str):
        inputs["text_g"] = prompt_text
    if isinstance(inputs.get("text_l"), str):
        inputs["text_l"] = prompt_text


def _apply_single_override(
    workflow: dict[str, dict[str, Any]],
    selector: str,
    value: Any,
) -> int:
    """単一のセレクタ指定でワークフローの入力値を上書きし、更新ノード数を返す。

    セレクタ書式:
    - `"6.text"` -> ノード ID `"6"` の `inputs["text"]`
    - `"title:Positive Prompt.text"` -> `_meta.title == "Positive Prompt"` の `inputs["text"]`
    - `"class:KSampler.steps"` -> `class_type == "KSampler"` の全ノードの `inputs["steps"]`
    """
    if "." not in selector:
        raise ValueError(
            f"上書きキー '{selector}' の書式が不正です。'<対象>.<フィールド名>' 形式で指定してください。"
        )

    target_spec, field_name = selector.rsplit(".", 1)
    if not target_spec or not field_name:
        raise ValueError(f"上書きキー '{selector}' の対象またはフィールド名が空です。")

    if target_spec.startswith("title:"):
        target_ids = find_nodes(workflow, title=target_spec[len("title:"):])
    elif target_spec.startswith("class:"):
        target_ids = find_nodes(workflow, class_type=target_spec[len("class:"):])
    else:
        target_ids = find_nodes(workflow, node_id=target_spec)

    if not target_ids:
        raise KeyError(f"上書き対象ノードが見つかりません: '{selector}'")

    for nid in target_ids:
        workflow[nid]["inputs"][field_name] = value

    return len(target_ids)


def load_preset(path: Path | str) -> dict[str, Any]:
    """SDXL 等の生成パラメータプリセット（YAML / JSON）を読み込む。"""
    preset_path = Path(path)
    if not preset_path.is_file():
        raise FileNotFoundError(f"プリセットファイルが見つかりません: {preset_path}")

    text = preset_path.read_text(encoding="utf-8")
    suffix = preset_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("YAML プリセットを読み込むには PyYAML が必要です: pip install PyYAML") from exc
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"未対応のプリセット形式です: {preset_path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"プリセットはオブジェクトである必要があります: {preset_path}")
    return data


def apply_sdxl_preset(
    workflow: Mapping[str, Mapping[str, Any]],
    preset: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """SDXL プリセット辞書（checkpoint / vae / 解像度 / KSampler / LoRA / プロンプト等）を適用する。

    ワークフロー内に該当ノードが存在する場合のみ安全に上書きし、
    値が None の項目はワークフロー側の元設定を維持する。
    """
    updated: dict[str, dict[str, Any]] = copy.deepcopy(dict(workflow))  # type: ignore[arg-type]

    def _set_on_classes(class_types: tuple[str, ...], field: str, value: Any) -> None:
        if value is None:
            return
        for node in updated.values():
            if node.get("class_type") in class_types:
                node.setdefault("inputs", {})[field] = value

    _set_on_classes(("CheckpointLoaderSimple", "CheckpointLoader"), "ckpt_name", preset.get("checkpoint"))
    _set_on_classes(("VAELoader",), "vae_name", preset.get("vae"))

    width = preset.get("width")
    height = preset.get("height")
    _set_on_classes(("EmptyLatentImage", "EmptySD3LatentImage"), "width", width)
    _set_on_classes(("EmptyLatentImage", "EmptySD3LatentImage"), "height", height)
    _set_on_classes(("EmptyLatentImage", "EmptySD3LatentImage"), "batch_size", preset.get("batch_size"))

    # CLIPTextEncodeSDXL を使用している場合は解像度メタデータも同期する
    if width is not None or height is not None:
        for node in updated.values():
            if node.get("class_type") == "CLIPTextEncodeSDXL":
                inputs = node.setdefault("inputs", {})
                if width is not None:
                    inputs["width"] = width
                    inputs["target_width"] = width
                if height is not None:
                    inputs["height"] = height
                    inputs["target_height"] = height

    sampler_classes = ("KSampler", "KSamplerAdvanced")
    _set_on_classes(sampler_classes, "steps", preset.get("steps"))
    _set_on_classes(sampler_classes, "cfg", preset.get("cfg"))
    _set_on_classes(sampler_classes, "sampler_name", preset.get("sampler_name"))
    _set_on_classes(sampler_classes, "scheduler", preset.get("scheduler"))
    _set_on_classes(("KSampler",), "denoise", preset.get("denoise"))

    lora_cfg = preset.get("lora")
    if isinstance(lora_cfg, Mapping):
        lora_classes = ("LoraLoader", "LoraLoaderModelOnly")
        _set_on_classes(lora_classes, "lora_name", lora_cfg.get("name"))
        _set_on_classes(lora_classes, "strength_model", lora_cfg.get("strength_model"))
        _set_on_classes(("LoraLoader",), "strength_clip", lora_cfg.get("strength_clip"))

    extra_overrides = preset.get("overrides")
    return apply_overrides(
        updated,
        overrides=extra_overrides if isinstance(extra_overrides, Mapping) else None,
        positive_prompt=preset.get("positive_prompt"),
        negative_prompt=preset.get("negative_prompt"),
        seed=preset.get("seed"),
        randomize_seed=bool(preset.get("randomize_seed", False)),
    )


def apply_overrides(
    workflow: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Any] | None = None,
    *,
    randomize_seed: bool = False,
    seed: int | None = None,
    positive_prompt: str | None = None,
    negative_prompt: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, dict[str, Any]]:
    """ワークフローをディープコピーし、プロンプト・シード・任意ノード入力を上書きする。

    Parameters
    ----------
    workflow : Mapping[str, Mapping[str, Any]]
        元の API 形式ワークフロー（変更されない）。
    overrides : Mapping[str, Any] | None
        `{"<selector>.<field>": value}` 形式の上書き辞書。
    randomize_seed : bool
        True の場合、シード入力（`seed` / `noise_seed`）をランダム値に差し替える。
    seed : int | None
        指定した場合、シード入力を固定値に設定する（`randomize_seed` より優先）。
    positive_prompt : str | None
        指定した場合、Positive プロンプト用テキストノードを上書きする。
    negative_prompt : str | None
        指定した場合、Negative プロンプト用テキストノードを上書きする。
    rng : random.Random | None
        テスト用の乱数生成器。

    Returns
    -------
    dict[str, dict[str, Any]]
        上書き後のワークフロー辞書。
    """
    updated: dict[str, dict[str, Any]] = copy.deepcopy(dict(workflow))  # type: ignore[arg-type]
    random_gen = rng or random

    if seed is not None or randomize_seed:
        for node in updated.values():
            inputs = node.get("inputs", {})
            for field in SEED_FIELD_NAMES:
                if field in inputs and isinstance(inputs[field], int) and not isinstance(inputs[field], bool):
                    inputs[field] = seed if seed is not None else random_gen.randint(0, 2**32 - 1)

    if positive_prompt is not None:
        pos_ids = _resolve_prompt_node_ids(updated, "positive")
        if not pos_ids:
            raise KeyError(
                "Positive プロンプト対象のノードを自動検出できませんでした。"
                "--set '<node_id>.text=...' で明示指定してください。"
            )
        for nid in pos_ids:
            _set_prompt_text(updated[nid], positive_prompt)

    if negative_prompt is not None:
        neg_ids = _resolve_prompt_node_ids(updated, "negative")
        if not neg_ids:
            raise KeyError(
                "Negative プロンプト対象のノードを自動検出できませんでした。"
                "--set '<node_id>.text=...' で明示指定してください。"
            )
        for nid in neg_ids:
            _set_prompt_text(updated[nid], negative_prompt)

    if overrides:
        for selector, value in overrides.items():
            _apply_single_override(updated, selector, value)

    return updated


def parse_override_spec(spec: str) -> tuple[str, Any]:
    """CLI の `--set key=value` 文字列を `(selector, parsed_value)` に変換する。"""
    if "=" not in spec:
        raise ValueError(
            f"--set の指定 '{spec}' に '=' が含まれていません。例: --set class:KSampler.steps=25"
        )
    key, raw_value = spec.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()

    # JSON として解釈できる値（数値・真偽値・配列・オブジェクト・null）は型変換する
    try:
        parsed_value: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed_value = raw_value

    return key, parsed_value


def collect_input_image_nodes(
    workflow: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """`LoadImage` 系ノードの `(node_id, image_value)` 一覧を返す。"""
    results: list[tuple[str, str]] = []
    for nid, node in workflow.items():
        if node.get("class_type") in LOAD_IMAGE_CLASS_TYPES:
            image_val = node.get("inputs", {}).get("image")
            if isinstance(image_val, str) and image_val.strip():
                results.append((nid, image_val.strip()))
    return results
