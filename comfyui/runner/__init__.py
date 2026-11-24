"""ComfyUI ワークフロー実行パッケージ。

将来の `colab/illustration-generator` 等からも import して利用できるよう、
クライアント・サーバー管理・ワークフロー操作の公開 API をまとめる。
"""

from .client import ComfyUIClient, ComfyUIError, RunResult
from .models import (
    MODEL_NODE_FIELD_MAP,
    ModelSpec,
    check_workflow_models,
    download_models,
    load_model_manifest,
)
from .prompt import GeneratedPrompt, PromptGenerator, PromptSlot
from .server import (
    DEFAULT_MODEL_SUBDIRS,
    ComfyUIServer,
    generate_extra_model_paths_yaml,
    link_drive_models,
)
from .workflow import (
    WorkflowFormatError,
    apply_overrides,
    apply_sdxl_preset,
    collect_input_image_nodes,
    find_nodes,
    load_preset,
    load_workflow,
    parse_override_spec,
    validate_api_workflow,
)

__all__ = [
    "ComfyUIClient",
    "ComfyUIError",
    "ComfyUIServer",
    "DEFAULT_MODEL_SUBDIRS",
    "GeneratedPrompt",
    "MODEL_NODE_FIELD_MAP",
    "ModelSpec",
    "PromptGenerator",
    "PromptSlot",
    "RunResult",
    "WorkflowFormatError",
    "apply_overrides",
    "apply_sdxl_preset",
    "check_workflow_models",
    "collect_input_image_nodes",
    "download_models",
    "find_nodes",
    "generate_extra_model_paths_yaml",
    "link_drive_models",
    "load_model_manifest",
    "load_preset",
    "load_workflow",
    "parse_override_spec",
    "validate_api_workflow",
]
