"""4-axis decision tree (Taxonomy) classifier and 1-pass inspection engine."""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .pose_variables import extract_pose_variables
from .tag_variables import (
    DEFAULT_IGNORED_IDENTITY_TAGS,
    DEFAULT_VARIABLE_TAG_SLOTS,
    extract_tag_variables,
)

DEFAULT_TAXONOMY_CONFIG: dict = {
    "framing_ambiguous_thr": 0.35,
    "camera_angle_thr": 0.30,
    # front は全件前向き（|shoulder_tilt_deg| <= 90°）のため前後分岐を行わない
    "split_body_facing_for": [
        "three_quarter_left",
        "three_quarter_right",
        "side_left",
        "side_right",
    ],
    # side_left / side_right の full_body は仰角変化がほぼ出ないため過分割を防止
    "split_camera_angle_for_full_body": [
        "front",
        "three_quarter_left",
        "three_quarter_right",
    ],
    # ambiguous（低スコア群）は除外せず camera_angle_zone 付きで救済枠として保持
    "split_camera_angle_for_ambiguous": True,
    # side_* で shoulder_zone (front/back_neg/back_pos) を使うか body_facing を使うか
    "body_axis_mode": "body_facing",  # "body_facing" または "shoulder_zone"
}

DEFAULT_INSPECTION_RULES: dict = {
    # DINOv2 一貫性・重複除外（葉ノード単位）
    "min_visual_centroid_sim": 0.60,
    "min_leaf_centroid_sim": 0.65,
    "min_anchor_sim": None,
    "max_leaf_pair_sim_dedup": 0.96,
    # rtmlib 骨格品質・プロポーション
    "require_pose_valid": True,
    "max_person_count": 1,
    "min_body_mean_score": 0.45,
    "proportion_z_max_front": 2.8,
    "proportion_z_max_other": 3.0,
    # 全画角（cowboy_shot / upper_body 含む）での前後整合チェック（暫定標準適用）
    "exclude_looking_back_pose_in_front_leaves": True,
    # WD14 共通必須 / 禁止タグ
    "required_tags": {
        "1girl": 0.70,
    },
    "forbidden_tags": {
        "monochrome": 0.50,
        "multiple_girls": 0.50,
    },
    # front 向き（顔の向き=front）向けの追加禁止タグ
    "front_facing_forbidden_tags": {
        "from_behind": 0.45,
        "profile": 0.45,
    },
    # 前向き体幹（body_facing=front）向けの追加禁止タグ
    "front_body_forbidden_tags": {
        "from_behind": 0.45,
    },
    # full_body / eye_level の純度ライン（俯瞰外れ値の足切り）
    "full_body_eye_level_min_score": 0.55,
    "full_body_eye_level_max_from_above": 0.35,
}


def classify_shoulder_zone(deg) -> str:
    """shoulder_tilt_deg を ±90° 境界で front / back_neg / back_pos の 3 領域に分類する。"""
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return "unknown"
    val = float(deg)
    if val < -90.0:
        return "back_neg"
    if val > 90.0:
        return "back_pos"
    return "front"


def discover_input_images(
    root_dir: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
) -> list[dict]:
    """指定ディレクトリ配下の画像を再帰的に収集する。"""
    root = Path(root_dir)
    if not root.exists():
        return []
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(root)
        rel_id = str(rel.with_suffix("")).replace("\\", "/")
        items.append(
            {
                "image_id": rel_id,
                "file_name": path.name,
                "file_path": str(path),
            }
        )
    return items


def assign_taxonomy_node(
    pose_vars: dict,
    tag_vars: dict,
    config: dict | None = None,
) -> dict:
    """
    4 軸決定木（facing_direction -> body_facing/shoulder_zone -> top_framing_tag -> camera_angle_zone）
    に従い、1 枚の画像の葉ノード ID（leaf_id）と各分岐ラベルを返す。
    """
    cfg = {**DEFAULT_TAXONOMY_CONFIG, **(config or {})}
    ori = pose_vars.get("orientation", {})
    dom = tag_vars.get("slot_dominant", {})
    cam = tag_vars.get("camera_angle", {})

    facing = ori.get("facing_direction") or "unknown"
    body_facing = ori.get("body_facing") or "unknown"
    shoulder_zone = classify_shoulder_zone(ori.get("shoulder_tilt_deg"))
    is_looking_back = bool(ori.get("is_looking_back_pose", False))

    framing_score = float(dom.get("framing", {}).get("score", 0.0) or 0.0)
    raw_framing_tag = dom.get("framing", {}).get("raw_top_tag") or dom.get("framing", {}).get("tag") or "ambiguous"
    framing_tag = (
        raw_framing_tag
        if framing_score >= float(cfg["framing_ambiguous_thr"])
        else "ambiguous"
    )

    from_above = float(cam.get("from_above_score", 0.0) or 0.0)
    from_below = float(cam.get("from_below_score", 0.0) or 0.0)
    cam_thr = float(cfg["camera_angle_thr"])
    if from_above >= cam_thr and from_above >= from_below:
        camera_angle_zone = "from_above"
    elif from_below >= cam_thr and from_below > from_above:
        camera_angle_zone = "from_below"
    else:
        camera_angle_zone = "eye_level"

    # --- 決定木パス構築 ---
    parts = [facing]

    # 第2分岐: side_* / three_quarter_* のみ体の前後向きで分岐（front は前向き固定のため分岐なし）
    if facing in cfg["split_body_facing_for"]:
        body_label = (
            shoulder_zone if cfg.get("body_axis_mode") == "shoulder_zone" else body_facing
        )
        parts.append(body_label)
    else:
        body_label = "front"

    # 第3分岐: 画角（top_framing_tag / ambiguous）
    parts.append(framing_tag)

    # 第4分岐: full_body（front / three_quarter_* のみ）および ambiguous は camera_angle_zone で分岐
    if (
        framing_tag == "full_body"
        and facing in cfg["split_camera_angle_for_full_body"]
    ):
        parts.append(camera_angle_zone)
    elif framing_tag == "ambiguous" and cfg.get("split_camera_angle_for_ambiguous", True):
        parts.append(camera_angle_zone)

    leaf_id = "/".join(parts)
    is_front_leaf = (facing == "front") or (body_label == "front")

    return {
        "leaf_id": leaf_id,
        "facing_direction": facing,
        "body_facing": body_facing,
        "shoulder_zone": shoulder_zone,
        "body_branch": body_label,
        "is_front_leaf": is_front_leaf,
        "is_looking_back_pose": is_looking_back,
        "top_framing_tag": framing_tag,
        "top_framing_score": round(framing_score, 4),
        "camera_angle_zone": camera_angle_zone,
    }


def run_taxonomy_inspection(
    image_items: list[dict],
    pose_model,
    tagger,
    embedder,
    taxonomy_config: dict | None = None,
    inspection_rules: dict | None = None,
    slot_definitions: dict[str, list[str]] | None = None,
    anchor_paths: list[str] | None = None,
    kpt_thr: float = 0.35,
    general_threshold: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    """
    1 パスの推論（rtmlib + WD14 + DINOv2）で全画像を決定木の葉ノード（leaf_id）へ自動分類し、
    葉ノード単位の DINOv2 一貫性・重複検知・プロポーション外れ値・タグ条件を一括評価する。
    """
    assert len(image_items) > 0, "入力画像が1枚も見つかりません。"

    cfg = {**DEFAULT_TAXONOMY_CONFIG, **(taxonomy_config or {})}
    rules = {**DEFAULT_INSPECTION_RULES, **(inspection_rules or {})}
    slots_cfg = (
        slot_definitions
        if slot_definitions is not None
        else DEFAULT_VARIABLE_TAG_SLOTS
    )

    records = []
    general_preds = []
    embeddings = []
    nodes = []

    for idx, item in enumerate(image_items, 1):
        img_bgr = cv2.imread(item["file_path"], cv2.IMREAD_UNCHANGED)
        assert img_bgr is not None, f"画像が読み込めません: {item['file_path']}"

        if img_bgr.ndim == 2:
            bgr_for_pose = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
            bgr_for_pose = (
                img_bgr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
            ).astype(np.uint8)
        else:
            bgr_for_pose = img_bgr

        h, w = bgr_for_pose.shape[:2]
        kpts, scores = pose_model(bgr_for_pose)
        pose_vars = extract_pose_variables(kpts, scores, bgr_for_pose.shape, kpt_thr=kpt_thr)
        wd14_pred = tagger.predict(img_bgr)
        tag_vars = extract_tag_variables(
            wd14_pred,
            slot_definitions=slots_cfg,
            general_threshold=general_threshold,
            slot_min_score=float(cfg["framing_ambiguous_thr"]),
        )
        node = assign_taxonomy_node(pose_vars, tag_vars, config=cfg)
        emb = embedder.embed(img_bgr)

        records.append(
            {
                **item,
                "meta": {"width": int(w), "height": int(h), "aspect_ratio": round(w / h, 4)},
                "pose": pose_vars,
                "tags": tag_vars,
                "taxonomy": node,
            }
        )
        general_preds.append(wd14_pred["general"])
        embeddings.append(emb)
        nodes.append(node)

        if idx % 20 == 0 or idx == len(image_items):
            print(f"[Inspect {idx}/{len(image_items)}] latest={item['image_id']} -> {node['leaf_id']}")

    # --- DINOv2 類似度（全体重心・葉ノード重心・葉ノード内最類似） ---
    emb_mat = np.vstack(embeddings).astype(np.float32)
    sim_matrix = np.clip(emb_mat @ emb_mat.T, -1.0, 1.0)
    n = len(records)

    global_c = emb_mat.mean(axis=0)
    global_c /= max(float(np.linalg.norm(global_c)), 1e-8)
    visual_centroid_sims = emb_mat @ global_c

    anchor_sims = None
    if anchor_paths:
        a_embs = []
        for ap in anchor_paths:
            a_img = cv2.imread(ap, cv2.IMREAD_UNCHANGED)
            if a_img is not None:
                a_embs.append(embedder.embed(a_img))
        if a_embs:
            a_c = np.vstack(a_embs).mean(axis=0)
            a_c /= max(float(np.linalg.norm(a_c)), 1e-8)
            anchor_sims = emb_mat @ a_c

    leaf_ids = [nd["leaf_id"] for nd in nodes]
    leaf_centroid_sims = np.zeros(n, dtype=np.float32)
    leaf_max_pair_sims = np.zeros(n, dtype=np.float32)
    leaf_nearest = [None] * n

    for leaf in sorted(set(leaf_ids)):
        idxs = [i for i, lf in enumerate(leaf_ids) if lf == leaf]
        sub_emb = emb_mat[idxs]
        lc = sub_emb.mean(axis=0)
        lc /= max(float(np.linalg.norm(lc)), 1e-8)
        leaf_centroid_sims[idxs] = sub_emb @ lc

        if len(idxs) > 1:
            sub_sim = sim_matrix[np.ix_(idxs, idxs)].copy()
            np.fill_diagonal(sub_sim, -1.0)
            best_local = np.argmax(sub_sim, axis=1)
            for pos, global_i in enumerate(idxs):
                j_local = int(best_local[pos])
                leaf_max_pair_sims[global_i] = float(sub_sim[pos, j_local])
                leaf_nearest[global_i] = records[idxs[j_local]]["image_id"]

    # --- フラットテーブル構築 ---
    rows = []
    for i, (rec, nd) in enumerate(zip(records, nodes)):
        pose = rec["pose"]
        dq = pose["detection_quality"]
        fr = pose["framing"]
        ori = pose["orientation"]
        pr = pose["proportions"]
        dom = rec["tags"]["slot_dominant"]
        cam = rec["tags"].get("camera_angle", {})
        gaze_slots = rec["tags"]["slots"].get("gaze", {})

        row = {
            "image_id": rec["image_id"],
            "file_name": rec["file_name"],
            "file_path": rec["file_path"],
            "leaf_id": nd["leaf_id"],
            "facing_direction": nd["facing_direction"],
            "body_facing": nd["body_facing"],
            "shoulder_zone": nd["shoulder_zone"],
            "is_front_leaf": nd["is_front_leaf"],
            "is_looking_back_pose": nd["is_looking_back_pose"],
            "top_framing_tag": nd["top_framing_tag"],
            "top_framing_score": nd["top_framing_score"],
            "estimated_shot": fr.get("estimated_shot"),
            "camera_angle_zone": nd["camera_angle_zone"],
            "from_above_score": cam.get("from_above_score"),
            "from_below_score": cam.get("from_below_score"),
            "camera_pitch_proxy": cam.get("camera_pitch_proxy"),
            "feet_out_of_frame_score": cam.get("feet_out_of_frame_score"),
            "visual_centroid_sim": round(float(visual_centroid_sims[i]), 4),
            "leaf_centroid_sim": round(float(leaf_centroid_sims[i]), 4),
            "leaf_max_pair_sim": round(float(leaf_max_pair_sims[i]), 4),
            "leaf_nearest_image": leaf_nearest[i],
            "pose_valid": dq["is_valid"],
            "person_count": dq["person_count"],
            "body_mean_score": dq["body_mean_score"],
            "head_yaw_proxy": ori.get("head_yaw_proxy"),
            "head_roll_deg": ori.get("head_roll_deg"),
            "shoulder_tilt_deg": ori.get("shoulder_tilt_deg"),
            "eye_to_shoulder_ratio": pr.get("eye_to_shoulder_ratio"),
            "torso_to_shoulder_ratio": pr.get("torso_to_shoulder_ratio"),
            "hip_to_shoulder_ratio": pr.get("hip_to_shoulder_ratio"),
            "top_expression": dom.get("expression", {}).get("tag"),
            "top_expression_score": dom.get("expression", {}).get("score"),
            "looking_back_score": gaze_slots.get("looking_back"),
            "from_behind_score": gaze_slots.get("from_behind"),
        }
        if anchor_sims is not None:
            row["anchor_sim"] = round(float(anchor_sims[i]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- 葉ノード内プロポーション統計 ---
    prop_cols = ["eye_to_shoulder_ratio", "torso_to_shoulder_ratio", "hip_to_shoulder_ratio"]
    leaf_prop_stats = {}
    for leaf, grp in df.groupby("leaf_id"):
        stats = {}
        for col in prop_cols:
            vals = grp[col].dropna()
            if len(vals) >= 4:
                stats[col] = (float(vals.mean()), max(float(vals.std(ddof=0)), 0.04))
        leaf_prop_stats[leaf] = stats

    # --- 決定木ルール評価 ---
    reasons_list = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
        r = reasons_list[i]
        pred = general_preds[i]
        facing = row["facing_direction"]
        leaf = row["leaf_id"]

        # 1. DINOv2
        if rules.get("min_visual_centroid_sim") is not None and row["visual_centroid_sim"] < rules["min_visual_centroid_sim"]:
            r.append(f"low_visual_sim({row['visual_centroid_sim']:.3f})")
        if rules.get("min_leaf_centroid_sim") is not None and row["leaf_centroid_sim"] < rules["min_leaf_centroid_sim"]:
            r.append(f"low_leaf_sim({row['leaf_centroid_sim']:.3f})")
        if rules.get("min_anchor_sim") is not None and "anchor_sim" in row and pd.notna(row["anchor_sim"]) and row["anchor_sim"] < rules["min_anchor_sim"]:
            r.append(f"low_anchor_sim({row['anchor_sim']:.3f})")

        # 2. rtmlib 骨格品質・前後整合・プロポーション
        if rules.get("require_pose_valid", True) and not bool(row["pose_valid"]):
            r.append("pose_invalid")
        if rules.get("max_person_count") is not None and row["person_count"] > rules["max_person_count"]:
            r.append(f"person_count({row['person_count']})")
        if rules.get("min_body_mean_score") is not None and row["body_mean_score"] < rules["min_body_mean_score"]:
            r.append(f"low_pose_score({row['body_mean_score']:.2f})")

        # 全画角での前後整合（前向き葉に見返りポーズが紛れた場合を除外）
        if rules.get("exclude_looking_back_pose_in_front_leaves", True) and bool(row["is_front_leaf"]) and bool(row["is_looking_back_pose"]):
            r.append("looking_back_pose")

        z_max = (
            rules.get("proportion_z_max_front", 2.8)
            if facing == "front"
            else rules.get("proportion_z_max_other", 3.0)
        )
        if z_max is not None:
            for col, (m, sd) in leaf_prop_stats.get(leaf, {}).items():
                if pd.notna(row[col]):
                    z = abs((float(row[col]) - m) / sd)
                    if z > z_max:
                        r.append(f"prop_outlier:{col}(z={z:.2f})")

        # 3. WD14 タグ条件（共通 + 向き/体幹/full_body_eye_level）
        for tag, min_sc in rules.get("required_tags", {}).items():
            actual = float(pred.get(tag, 0.0))
            if actual < min_sc:
                r.append(f"missing_tag:{tag}({actual:.2f})")

        forb_tags = dict(rules.get("forbidden_tags", {}))
        if facing == "front":
            forb_tags.update(rules.get("front_facing_forbidden_tags", {}))
        if bool(row["is_front_leaf"]):
            forb_tags.update(rules.get("front_body_forbidden_tags", {}))

        for tag, max_sc in forb_tags.items():
            actual = float(pred.get(tag, 0.0))
            if actual > max_sc:
                r.append(f"forbidden_tag:{tag}({actual:.2f})")

        # full_body / eye_level の純度チェック（俯瞰外れ値の足切り）
        if row["top_framing_tag"] == "full_body" and row["camera_angle_zone"] == "eye_level":
            fb_min = rules.get("full_body_eye_level_min_score")
            if fb_min is not None and row["top_framing_score"] < fb_min:
                r.append(f"low_full_body_score({row['top_framing_score']:.2f}<{fb_min})")
            fb_above = rules.get("full_body_eye_level_max_from_above")
            if fb_above is not None and float(row["from_above_score"] or 0.0) > fb_above:
                r.append(f"full_body_from_above({row['from_above_score']:.2f}>{fb_above})")

    # 4. 葉ノード内の重複除外（leaf_centroid_sim が高い方を優先保持）
    dedup_thr = rules.get("max_leaf_pair_sim_dedup")
    if dedup_thr is not None:
        for leaf in sorted(set(leaf_ids)):
            pass_idxs = [i for i in range(n) if leaf_ids[i] == leaf and len(reasons_list[i]) == 0]
            pass_idxs.sort(key=lambda i: float(df.loc[i, "leaf_centroid_sim"]), reverse=True)
            kept = []
            for idx in pass_idxs:
                dup_of = None
                for k in kept:
                    if sim_matrix[idx, k] >= dedup_thr:
                        dup_of = (df.loc[k, "image_id"], float(sim_matrix[idx, k]))
                        break
                if dup_of is not None:
                    reasons_list[idx].append(f"near_duplicate:{dup_of[0]}({dup_of[1]:.3f})")
                else:
                    kept.append(idx)

    df["selected"] = [len(r) == 0 for r in reasons_list]
    df["reject_reasons"] = ["; ".join(r) for r in reasons_list]

    leaf_summary_df = (
        df.groupby("leaf_id")
        .agg(
            total=("selected", "count"),
            selected=("selected", "sum"),
            mean_leaf_sim=("leaf_centroid_sim", "mean"),
            mean_framing_score=("top_framing_score", "mean"),
        )
        .assign(
            rejected=lambda d: d["total"] - d["selected"],
            accept_rate=lambda d: (d["selected"] / d["total"]).round(3),
        )
        .round(3)
        .sort_values("total", ascending=False)
    )
    return df, leaf_summary_df, records, general_preds


def export_taxonomy_results(
    result_df: pd.DataFrame,
    leaf_summary_df: pd.DataFrame,
    output_dir: str,
    taxonomy_config: dict | None = None,
    inspection_rules: dict | None = None,
    clear_export_dirs: bool = True,
) -> dict:
    """
    決定木検査の結果を書き出す:
    - selected/<leaf_id>/<file_name>
    - rejected/<reason>/<file_name> + rejected/rejected_list.csv
    - inspection_metrics.csv / leaf_summary.csv / taxonomy_config.json
    """
    os.makedirs(output_dir, exist_ok=True)
    sel_dir = os.path.join(output_dir, "selected")
    rej_dir = os.path.join(output_dir, "rejected")

    for d in (sel_dir, rej_dir):
        if clear_export_dirs and os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    sel_df = result_df[result_df["selected"]]
    for _, row in sel_df.iterrows():
        dst_dir = os.path.join(sel_dir, str(row["leaf_id"]))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(row["file_path"], os.path.join(dst_dir, row["file_name"]))

    rej_df = result_df[~result_df["selected"]]
    for _, row in rej_df.iterrows():
        first_reason = str(row["reject_reasons"]).split("; ")[0].split("(")[0].split(":")[0] or "rejected"
        dst_dir = os.path.join(rej_dir, first_reason.strip().replace("/", "_"))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(row["file_path"], os.path.join(dst_dir, row["file_name"]))

    metrics_csv = os.path.join(output_dir, "inspection_metrics.csv")
    summary_csv = os.path.join(output_dir, "leaf_summary.csv")
    rej_csv = os.path.join(rej_dir, "rejected_list.csv")
    cfg_json = os.path.join(output_dir, "taxonomy_config.json")

    result_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    leaf_summary_df.to_csv(summary_csv, encoding="utf-8-sig")
    rej_df.to_csv(rej_csv, index=False, encoding="utf-8-sig")

    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "taxonomy_config": taxonomy_config or DEFAULT_TAXONOMY_CONFIG,
                "inspection_rules": inspection_rules or DEFAULT_INSPECTION_RULES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ 採用画像 {len(sel_df)} 枚 -> {sel_dir} ({sel_df['leaf_id'].nunique()} 葉ノード)")
    print(f"🗂️ 除外画像 {len(rej_df)} 枚 -> {rej_dir}")
    print(f"📄 レポート保存 -> {metrics_csv}, {summary_csv}, {cfg_json}")
    return {
        "selected_dir": sel_dir,
        "rejected_dir": rej_dir,
        "metrics_csv": metrics_csv,
        "summary_csv": summary_csv,
        "config_json": cfg_json,
    }
