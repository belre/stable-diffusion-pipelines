"""Shared utilities for colab/snapshot-inspector character inspection notebooks."""

from .audit import (
    SPECIAL_FRAMING_LEAVES,
    apply_relabel_moves,
    discover_labeled_dataset,
    extract_or_load_dataset_features,
    parse_labeled_path,
    run_dataset_audit,
    show_audit_suspects,
)
from .models import (
    DEFAULT_DINOV2_REPO,
    DEFAULT_WD14_REPO,
    DINOv2Embedder,
    WD14Tagger,
    get_onnx_device_config,
    init_wholebody,
)
from .pose_variables import (
    COCO_BODY_NAMES,
    KP,
    extract_pose_variables,
    round_or_none,
)
from .profile import (
    build_character_profile,
    evaluate_against_profile,
    extract_image_variables,
    list_image_files,
    variables_to_flat_row,
)
from .tag_variables import (
    DEFAULT_IGNORED_IDENTITY_TAGS,
    DEFAULT_VARIABLE_TAG_SLOTS,
    extract_tag_variables,
)
from .taxonomy import (
    DEFAULT_INSPECTION_RULES,
    DEFAULT_TAXONOMY_CONFIG,
    assign_taxonomy_node,
    classify_shoulder_zone,
    discover_input_images,
    export_taxonomy_results,
    run_taxonomy_inspection,
)
from .visualization import (
    archive_folder_to_drive,
    save_results_to_drive,
    show_batch_variable_heatmap,
    show_leaf_summary_chart,
    show_leaf_thumbnails,
    show_profile_summary,
    show_target_report,
)

__all__ = [
    "COCO_BODY_NAMES",
    "DEFAULT_DINOV2_REPO",
    "DEFAULT_IGNORED_IDENTITY_TAGS",
    "DEFAULT_INSPECTION_RULES",
    "DEFAULT_TAXONOMY_CONFIG",
    "DEFAULT_VARIABLE_TAG_SLOTS",
    "DEFAULT_WD14_REPO",
    "DINOv2Embedder",
    "KP",
    "SPECIAL_FRAMING_LEAVES",
    "WD14Tagger",
    "apply_relabel_moves",
    "archive_folder_to_drive",
    "assign_taxonomy_node",
    "build_character_profile",
    "classify_shoulder_zone",
    "discover_input_images",
    "discover_labeled_dataset",
    "evaluate_against_profile",
    "export_taxonomy_results",
    "extract_image_variables",
    "extract_or_load_dataset_features",
    "extract_pose_variables",
    "extract_tag_variables",
    "get_onnx_device_config",
    "init_wholebody",
    "list_image_files",
    "parse_labeled_path",
    "round_or_none",
    "run_dataset_audit",
    "run_taxonomy_inspection",
    "save_results_to_drive",
    "show_audit_suspects",
    "show_batch_variable_heatmap",
    "show_leaf_summary_chart",
    "show_leaf_thumbnails",
    "show_profile_summary",
    "show_target_report",
    "variables_to_flat_row",
]
