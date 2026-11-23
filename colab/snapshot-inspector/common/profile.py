"""Image variable extraction, character distribution profiling, and consistency evaluation."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import cv2
import numpy as np

from .pose_variables import extract_pose_variables, round_or_none
from .tag_variables import (
    DEFAULT_IGNORED_IDENTITY_TAGS,
    DEFAULT_VARIABLE_TAG_SLOTS,
    extract_tag_variables,
)


def _summarize_numeric_distribution(values: list) -> dict | None:
    """数値リストから平均・標準偏差・中央値・最小・最大・件数を集計する。"""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=0)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def list_image_files(
    directory: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
) -> list[str]:
    """指定ディレクトリ内の画像ファイルをソート済みパス一覧で返す。"""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith(extensions)
    ]


def extract_image_variables(
    img_path: str,
    pose_model,
    tagger,
    slot_definitions: dict[str, list[str]] | None = None,
    kpt_thr: float = 0.35,
    general_threshold: float = 0.35,
    return_debug_artifacts: bool = False,
):
    """
    1枚の画像を読み込み、Pose（骨格）と WD14-Tagger（タグ信頼度）の両面から変数化する。
    """
    slots_cfg = (
        slot_definitions
        if slot_definitions is not None
        else DEFAULT_VARIABLE_TAG_SLOTS
    )

    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert img_bgr is not None, f"画像が読み込めません: {img_path}"

    # アルファチャンネル付き立ち絵 PNG の場合は白背景合成した BGR を骨格検出へ渡す
    if img_bgr.ndim == 2:
        bgr_for_pose = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img_bgr[:, :, :3].astype(np.float32)
        bgr_for_pose = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    else:
        bgr_for_pose = img_bgr

    h, w = bgr_for_pose.shape[:2]

    # 1. 骨格変数
    kpts, scores = pose_model(bgr_for_pose)
    pose_vars = extract_pose_variables(kpts, scores, bgr_for_pose.shape, kpt_thr=kpt_thr)

    # 2. WD14-Tagger 変数
    wd14_pred = tagger.predict(img_bgr)
    tag_vars = extract_tag_variables(
        wd14_pred,
        slot_definitions=slots_cfg,
        general_threshold=general_threshold,
    )

    record = {
        "image_id": os.path.splitext(os.path.basename(img_path))[0],
        "file_name": os.path.basename(img_path),
        "file_path": img_path,
        "meta": {
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(w / h, 4),
            "has_alpha": bool(img_bgr.ndim == 3 and img_bgr.shape[2] == 4),
        },
        "pose": pose_vars,
        "tags": tag_vars,
    }

    if return_debug_artifacts:
        from rtmlib import draw_skeleton

        debug_skel = draw_skeleton(bgr_for_pose.copy(), kpts, scores, kpt_thr=kpt_thr)
        return record, wd14_pred, bgr_for_pose, debug_skel
    return record, wd14_pred


def build_character_profile(
    reference_paths: list[str],
    pose_model,
    tagger,
    character_name: str = "character_default",
    slot_definitions: dict[str, list[str]] | None = None,
    ignored_identity_tags: set[str] | None = None,
    force_identity_tags: list[str] | None = None,
    exclude_identity_tags: list[str] | None = None,
    core_presence_threshold: float = 0.65,
    core_mean_threshold: float = 0.45,
    tag_active_threshold: float = 0.35,
    kpt_thr: float = 0.35,
) -> tuple[dict, list[dict]]:
    """
    複数の LoRA 学習データ／リファレンス画像群を解析し、
    キャラクター固有の特徴分布（Character Profile）を構築する。
    """
    assert len(reference_paths) > 0, "リファレンス画像が1枚も見つかりません。"

    slots_cfg = (
        slot_definitions
        if slot_definitions is not None
        else DEFAULT_VARIABLE_TAG_SLOTS
    )
    ignored_set = (
        ignored_identity_tags
        if ignored_identity_tags is not None
        else DEFAULT_IGNORED_IDENTITY_TAGS
    )
    force_set = set(force_identity_tags or [])
    exclude_set = set(exclude_identity_tags or [])
    variable_tag_set = {t for tags in slots_cfg.values() for t in tags}

    records = []
    all_general_preds = []

    for idx, path in enumerate(reference_paths, 1):
        rec, wd14_pred = extract_image_variables(
            path,
            pose_model=pose_model,
            tagger=tagger,
            slot_definitions=slots_cfg,
            kpt_thr=kpt_thr,
            general_threshold=tag_active_threshold,
        )
        records.append(rec)
        all_general_preds.append(wd14_pred["general"])
        print(
            f"[Profile {idx}/{len(reference_paths)}] {rec['file_name']} "
            f"(pose_valid={rec['pose']['detection_quality']['is_valid']}, "
            f"shot={rec['pose']['framing'].get('estimated_shot')})"
        )

    n_imgs = len(records)
    all_tag_names = list(all_general_preds[0].keys())

    # 1. キャラ固有の不変特徴タグ（identity_core_tags）の統計分布を自動同定
    identity_core_tags = {}
    reference_general_mean = {}

    for tag in all_tag_names:
        vals = np.array([pred[tag] for pred in all_general_preds], dtype=np.float64)
        mean_val = float(np.mean(vals))
        max_val = float(np.max(vals))
        if max_val >= 0.15:
            reference_general_mean[tag] = round(mean_val, 4)

        if tag in exclude_set:
            continue
        is_forced = tag in force_set
        if not is_forced and (tag in variable_tag_set or tag in ignored_set):
            continue

        presence_rate = float(np.mean(vals >= tag_active_threshold))
        if is_forced or (
            presence_rate >= core_presence_threshold and mean_val >= core_mean_threshold
        ):
            identity_core_tags[tag] = {
                "mean": round(mean_val, 4),
                "std": round(float(np.std(vals, ddof=0)), 4),
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "presence_rate": round(presence_rate, 4),
                "forced": is_forced,
            }

    identity_core_tags = dict(
        sorted(identity_core_tags.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    )

    # 2. スケール不変骨格プロポーションの分布
    proportion_keys = [
        "eye_to_shoulder_ratio",
        "torso_to_shoulder_ratio",
        "hip_to_shoulder_ratio",
        "upper_arm_to_shoulder_ratio",
    ]
    proportions_dist = {}
    for key in proportion_keys:
        vals = [
            r["pose"]["proportions"].get(key)
            for r in records
            if r["pose"]["detection_quality"]["is_valid"]
        ]
        summary = _summarize_numeric_distribution(vals)
        if summary is not None:
            proportions_dist[key] = summary

    # 3. 可変スロットのベースライン分布
    variable_slots_baseline = {}
    for slot_name, tag_list in slots_cfg.items():
        slot_stats = {}
        for tag in tag_list:
            vals = [r["tags"]["slots"][slot_name][tag] for r in records]
            slot_stats[tag] = _summarize_numeric_distribution(vals)
        variable_slots_baseline[slot_name] = slot_stats

    # 4. 構図・向きのベースライン分布
    framing_baseline = {}
    for key in [
        "bbox_w_ratio",
        "bbox_h_ratio",
        "center_x",
        "center_y",
        "shoulder_width_ratio",
        "eye_dist_ratio",
    ]:
        vals = [
            r["pose"]["framing"].get(key)
            for r in records
            if r["pose"]["detection_quality"]["is_valid"]
        ]
        summary = _summarize_numeric_distribution(vals)
        if summary is not None:
            framing_baseline[key] = summary

    profile = {
        "schema_version": "1.0",
        "character_name": character_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wd14_model": getattr(tagger, "repo_id", "unknown"),
        "reference_count": n_imgs,
        "reference_files": [r["file_name"] for r in records],
        "thresholds": {
            "core_presence_threshold": core_presence_threshold,
            "core_mean_threshold": core_mean_threshold,
            "tag_active_threshold": tag_active_threshold,
            "kpt_thr": kpt_thr,
        },
        "identity_core_tags": identity_core_tags,
        "proportions": proportions_dist,
        "framing_baseline": framing_baseline,
        "variable_slots_baseline": variable_slots_baseline,
        "reference_general_mean": reference_general_mean,
    }
    return profile, records


def evaluate_against_profile(
    record: dict,
    wd14_pred: dict,
    profile: dict,
    slot_definitions: dict[str, list[str]] | None = None,
    ignored_identity_tags: set[str] | None = None,
    identity_z_warn: float = 2.0,
    identity_min_std: float = 0.08,
    proportion_z_warn: float = 2.5,
    proportion_min_std: float = 0.04,
    unexpected_tag_threshold: float = 0.55,
    unexpected_ref_max_mean: float = 0.15,
) -> dict:
    """
    対象画像の変数化結果をキャラクター固有分布（Character Profile）と照合し、
    一貫性スコアおよび逸脱アラートを算出する。
    """
    slots_cfg = (
        slot_definitions
        if slot_definitions is not None
        else DEFAULT_VARIABLE_TAG_SLOTS
    )
    ignored_set = (
        ignored_identity_tags
        if ignored_identity_tags is not None
        else DEFAULT_IGNORED_IDENTITY_TAGS
    )
    general_scores = wd14_pred["general"]
    core_tags = profile.get("identity_core_tags", {})
    ref_general_mean = profile.get("reference_general_mean", {})
    variable_tag_set = {t for tags in slots_cfg.values() for t in tags}

    # --- 1. アイデンティティ一貫性 ---
    core_details = {}
    missing_or_weak_traits = []
    weighted_match_sum = 0.0
    weight_total = 0.0

    for tag, stats in core_tags.items():
        actual = float(general_scores.get(tag, 0.0))
        mean = float(stats["mean"])
        std = float(stats["std"])
        eff_std = max(std, identity_min_std)
        z_score = (actual - mean) / eff_std

        shortfall = max(0.0, mean - actual)
        match = max(0.0, 1.0 - shortfall / max(mean, 1e-6))
        weighted_match_sum += mean * match
        weight_total += mean

        is_alert = (z_score <= -identity_z_warn) and (shortfall >= 0.20)
        core_details[tag] = {
            "actual": round(actual, 4),
            "expected_mean": round(mean, 4),
            "expected_std": round(std, 4),
            "z_score": round(z_score, 2),
            "alert": is_alert,
        }
        if is_alert:
            missing_or_weak_traits.append(
                {
                    "tag": tag,
                    "actual": round(actual, 4),
                    "expected_mean": round(mean, 4),
                    "expected_std": round(std, 4),
                    "z_score": round(z_score, 2),
                }
            )

    identity_score = (weighted_match_sum / weight_total) if weight_total > 0 else None

    unexpected_traits = []
    for tag, score in sorted(
        general_scores.items(), key=lambda kv: kv[1], reverse=True
    ):
        if score < unexpected_tag_threshold:
            break
        if tag in core_tags or tag in variable_tag_set or tag in ignored_set:
            continue
        ref_mean = float(ref_general_mean.get(tag, 0.0))
        if ref_mean <= unexpected_ref_max_mean:
            unexpected_traits.append(
                {
                    "tag": tag,
                    "actual": round(float(score), 4),
                    "reference_mean": round(ref_mean, 4),
                }
            )

    # --- 2. 骨格プロポーション一貫性 ---
    prop_profile = profile.get("proportions", {})
    prop_actual = record["pose"].get("proportions", {})
    prop_details = {}
    prop_alerts = []
    prop_scores = []

    for key, stats in prop_profile.items():
        actual = prop_actual.get(key)
        if actual is None:
            continue
        mean = float(stats["mean"])
        std = float(stats["std"])
        eff_std = max(std, proportion_min_std)
        z_score = (float(actual) - mean) / eff_std
        is_alert = abs(z_score) >= proportion_z_warn
        prop_scores.append(float(math.exp(-0.5 * (z_score / proportion_z_warn) ** 2)))
        prop_details[key] = {
            "actual": round(float(actual), 4),
            "expected_mean": round(mean, 4),
            "expected_std": round(std, 4),
            "z_score": round(z_score, 2),
            "alert": is_alert,
        }
        if is_alert:
            prop_alerts.append(
                {
                    "metric": key,
                    "actual": round(float(actual), 4),
                    "expected_mean": round(mean, 4),
                    "z_score": round(z_score, 2),
                }
            )

    proportion_score = float(np.mean(prop_scores)) if prop_scores else None

    # --- 3. 可変スロットのベースライン差分 ---
    slot_deltas = {}
    baseline_slots = profile.get("variable_slots_baseline", {})
    for slot_name, slot_vals in record["tags"]["slots"].items():
        base = baseline_slots.get(slot_name, {})
        deltas = {}
        for tag, val in slot_vals.items():
            base_mean = base.get(tag, {}).get("mean", 0.0) if base.get(tag) else 0.0
            deltas[tag] = round(float(val) - float(base_mean), 4)
        slot_deltas[slot_name] = deltas

    if identity_score is not None and proportion_score is not None:
        overall_score = 0.7 * identity_score + 0.3 * proportion_score
    else:
        overall_score = (
            identity_score if identity_score is not None else proportion_score
        )

    return {
        "character_name": profile.get("character_name"),
        "overall_consistency": round_or_none(overall_score),
        "identity_consistency": round_or_none(identity_score),
        "proportion_consistency": round_or_none(proportion_score),
        "missing_or_weak_traits": missing_or_weak_traits,
        "unexpected_traits": unexpected_traits[:10],
        "proportion_alerts": prop_alerts,
        "core_tag_comparison": core_details,
        "proportion_comparison": prop_details,
        "variable_slot_deltas": slot_deltas,
    }


def variables_to_flat_row(record: dict, consistency: dict | None = None) -> dict:
    """
    一覧表（DataFrame / CSV）用に、画像変数と一貫性評価を1行のフラット辞書へ整形する。
    """
    pose = record["pose"]
    framing = pose.get("framing", {})
    orient = pose.get("orientation", {})
    props = pose.get("proportions", {})
    angles = pose.get("joint_angles", {})
    dom = record["tags"].get("slot_dominant", {})

    row = {
        "image_id": record["image_id"],
        "file_name": record["file_name"],
        "width": record["meta"]["width"],
        "height": record["meta"]["height"],
        "pose_valid": pose["detection_quality"]["is_valid"],
        "shot_pose": framing.get("estimated_shot"),
        "shot_tag": dom.get("framing", {}).get("tag"),
        "facing": orient.get("facing_direction"),
        "head_yaw": orient.get("head_yaw_proxy"),
        "head_roll_deg": orient.get("head_roll_deg"),
        "shoulder_tilt_deg": orient.get("shoulder_tilt_deg"),
        "center_x": framing.get("center_x"),
        "center_y": framing.get("center_y"),
        "eye_to_shoulder": props.get("eye_to_shoulder_ratio"),
        "torso_to_shoulder": props.get("torso_to_shoulder_ratio"),
        "l_elbow_deg": angles.get("left_elbow_deg"),
        "r_elbow_deg": angles.get("right_elbow_deg"),
        "top_expression": dom.get("expression", {}).get("tag"),
        "top_expression_score": dom.get("expression", {}).get("score"),
        "top_gaze": dom.get("gaze", {}).get("tag"),
        "top_gaze_score": dom.get("gaze", {}).get("score"),
        "top_pose_tag": dom.get("pose_action", {}).get("tag"),
        "top_pose_score": dom.get("pose_action", {}).get("score"),
    }

    if consistency is not None:
        row.update(
            {
                "overall_consistency": consistency.get("overall_consistency"),
                "identity_consistency": consistency.get("identity_consistency"),
                "proportion_consistency": consistency.get("proportion_consistency"),
                "weak_trait_count": len(consistency.get("missing_or_weak_traits", [])),
                "unexpected_trait_count": len(consistency.get("unexpected_traits", [])),
                "prop_alert_count": len(consistency.get("proportion_alerts", [])),
            }
        )
    return row
