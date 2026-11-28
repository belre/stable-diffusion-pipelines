"""3-tier labeled dataset audit engine (OOF classifier label-noise detection + outlier/duplicate checks)."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pose_variables import extract_pose_variables
from .tag_variables import DEFAULT_VARIABLE_TAG_SLOTS, extract_tag_variables
from .taxonomy import classify_shoulder_zone

SPECIAL_FRAMING_LEAVES = {"lower_body", "broken_anatomy"}


def parse_labeled_path(rel_path: Path) -> dict:
    """
    3 階層フォルダ構成 `向き / 画角 / アングル(__姿勢) / file` をパースして各軸ラベルを返す。
    例外構造（`front/lower_body/x.png`, `front/broken_anatomy/x.png`, `__other/x.png` 等）も安全に処理する。
    """
    parts = rel_path.parts[:-1]  # ファイル名を除いたフォルダ階層
    orientation = parts[0] if len(parts) >= 1 else "unknown"
    framing = parts[1] if len(parts) >= 2 else "none"
    angle_posture = parts[2] if len(parts) >= 3 else "none"

    if angle_posture == "none":
        camera_angle = "none"
        posture = "normal"
    elif "__" in angle_posture:
        camera_angle, posture = angle_posture.split("__", 1)
    else:
        camera_angle = angle_posture
        posture = "normal"

    leaf_path = "/".join(parts) if parts else "root"
    return {
        "leaf_path": leaf_path,
        "label_orientation": orientation,
        "label_framing": framing,
        "label_camera_angle": camera_angle,
        "label_posture": posture,
        "label_angle_posture": angle_posture,
    }


def discover_labeled_dataset(
    root_dir: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
) -> pd.DataFrame:
    """3 階層フィルタリング済みフォルダを再帰スキャンし、ラベル付きテーブルを返す。"""
    root = Path(root_dir)
    assert root.exists(), f"ディレクトリが見つかりません: {root_dir}"

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(root)
        parsed = parse_labeled_path(rel)
        # 内容ハッシュ（完全同一ファイルの重複配置検知用）
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        rows.append(
            {
                "image_id": str(rel.with_suffix("")).replace("\\", "/"),
                "file_name": path.name,
                "file_path": str(path),
                "rel_path": str(rel).replace("\\", "/"),
                "md5": md5,
                **parsed,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        dup_counts = df["md5"].value_counts()
        df["exact_dup_count"] = df["md5"].map(dup_counts).astype(int)
    return df


def extract_or_load_dataset_features(
    dataset_df: pd.DataFrame,
    cache_dir: str,
    pose_model=None,
    tagger=None,
    embedder=None,
    force_recompute: bool = False,
    kpt_thr: float = 0.35,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    全画像から rtmlib + WD14 + DINOv2 特徴量を抽出し、`cache_dir` にキャッシュ保存する。
    キャッシュと現在のファイル一覧（md5）が一致すれば 1 秒で再読み込みする。

    Returns
    -------
    feat_df : pd.DataFrame
        ラベル列 + 抽出された骨格・WD14 メトリクス列
    dino_embs : np.ndarray (N, D)
        L2 正規化済み DINOv2 埋め込み行列
    aux_feats : np.ndarray (N, M)
        骨格数値特徴量 + WD14 主要スロット信頼度ベクトル（Classifier 補助特徴量）
    aux_names : list[str]
        aux_feats の各列名
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_npz = os.path.join(cache_dir, "features_cache.npz")
    cache_csv = os.path.join(cache_dir, "features_table.csv")

    if not force_recompute and os.path.exists(cache_npz) and os.path.exists(cache_csv):
        cached_df = pd.read_csv(cache_csv)
        # md5 集合が完全一致していればキャッシュを再利用（フォルダ移動後のラベル変化は最新 dataset_df で上書き）
        if set(dataset_df["md5"]).issubset(set(cached_df["md5"])):
            data = np.load(cache_npz, allow_pickle=True)
            cached_md5_to_idx = {m: i for i, m in enumerate(cached_df["md5"].tolist())}
            order = [cached_md5_to_idx[m] for m in dataset_df["md5"].tolist()]

            dino_embs = data["dino_embs"][order]
            aux_feats = data["aux_feats"][order]
            aux_names = data["aux_names"].tolist()

            metric_cols = [c for c in cached_df.columns if c not in dataset_df.columns]
            reordered_metrics = cached_df.iloc[order][metric_cols].reset_index(drop=True)
            feat_df = pd.concat([dataset_df.reset_index(drop=True), reordered_metrics], axis=1)
            print(f"✅ キャッシュから特徴量を読み込みました ({len(feat_df)} 枚) -> {cache_npz}")
            return feat_df, dino_embs, aux_feats, aux_names

    assert pose_model is not None and tagger is not None and embedder is not None, (
        "初回抽出には pose_model, tagger, embedder が必要です。"
    )

    slot_tag_list = [
        (slot, tag)
        for slot, tags in DEFAULT_VARIABLE_TAG_SLOTS.items()
        for tag in tags
    ]
    extra_wd14_tags = [
        "feet_out_of_frame",
        "lower_body",
        "squatting",
        "lying",
        "on_back",
        "on_side",
        "back",
        "facing_away",
        "looking_back",
        "from_behind",
        "from_side",
        "profile",
        "from_above",
        "from_below",
    ]
    # 重複を除いた WD14 補助特徴量タグ一覧
    wd14_aux_tags = list(dict.fromkeys([t for _, t in slot_tag_list] + extra_wd14_tags))
    pose_num_cols = [
        "body_mean_score",
        "body_valid_ratio",
        "bbox_w_ratio",
        "bbox_h_ratio",
        "center_x",
        "center_y",
        "shoulder_width_ratio",
        "eye_dist_ratio",
        "head_yaw_proxy",
        "head_roll_deg",
        "shoulder_tilt_deg",
        "torso_tilt_deg",
        "eye_to_shoulder_ratio",
        "torso_to_shoulder_ratio",
        "hip_to_shoulder_ratio",
        "upper_arm_to_shoulder_ratio",
        "left_elbow_deg",
        "right_elbow_deg",
        "left_knee_deg",
        "right_knee_deg",
    ]
    aux_names = pose_num_cols + [f"wd14:{t}" for t in wd14_aux_tags]

    metric_rows = []
    dino_list = []
    aux_list = []

    total = len(dataset_df)
    for idx, row in dataset_df.reset_index(drop=True).iterrows():
        img_bgr = cv2.imread(row["file_path"], cv2.IMREAD_UNCHANGED)
        assert img_bgr is not None, f"画像が読み込めません: {row['file_path']}"

        if img_bgr.ndim == 2:
            bgr_for_pose = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
            bgr_for_pose = (
                img_bgr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
            ).astype(np.uint8)
        else:
            bgr_for_pose = img_bgr

        kpts, scores = pose_model(bgr_for_pose)
        pose_vars = extract_pose_variables(kpts, scores, bgr_for_pose.shape, kpt_thr=kpt_thr)
        wd14_pred = tagger.predict(img_bgr)
        tag_vars = extract_tag_variables(wd14_pred, slot_definitions=DEFAULT_VARIABLE_TAG_SLOTS)
        emb = embedder.embed(img_bgr)

        dq = pose_vars["detection_quality"]
        fr = pose_vars["framing"]
        ori = pose_vars["orientation"]
        pr = pose_vars["proportions"]
        ja = pose_vars["joint_angles"]
        cam = tag_vars["camera_angle"]
        dom = tag_vars["slot_dominant"]
        gen = wd14_pred["general"]

        m_row = {
            "pose_valid": dq["is_valid"],
            "person_count": dq["person_count"],
            "body_mean_score": dq["body_mean_score"],
            "body_valid_ratio": dq["body_valid_ratio"],
            "estimated_shot": fr.get("estimated_shot"),
            "pred_facing_rule": ori.get("facing_direction"),
            "body_facing": ori.get("body_facing"),
            "shoulder_zone": classify_shoulder_zone(ori.get("shoulder_tilt_deg")),
            "is_looking_back_pose": ori.get("is_looking_back_pose"),
            "bbox_w_ratio": fr.get("bbox_w_ratio"),
            "bbox_h_ratio": fr.get("bbox_h_ratio"),
            "center_x": fr.get("center_x"),
            "center_y": fr.get("center_y"),
            "shoulder_width_ratio": fr.get("shoulder_width_ratio"),
            "eye_dist_ratio": fr.get("eye_dist_ratio"),
            "head_yaw_proxy": ori.get("head_yaw_proxy"),
            "head_roll_deg": ori.get("head_roll_deg"),
            "shoulder_tilt_deg": ori.get("shoulder_tilt_deg"),
            "torso_tilt_deg": ori.get("torso_tilt_deg"),
            "eye_to_shoulder_ratio": pr.get("eye_to_shoulder_ratio"),
            "torso_to_shoulder_ratio": pr.get("torso_to_shoulder_ratio"),
            "hip_to_shoulder_ratio": pr.get("hip_to_shoulder_ratio"),
            "upper_arm_to_shoulder_ratio": pr.get("upper_arm_to_shoulder_ratio"),
            "left_elbow_deg": ja.get("left_elbow_deg"),
            "right_elbow_deg": ja.get("right_elbow_deg"),
            "left_knee_deg": ja.get("left_knee_deg"),
            "right_knee_deg": ja.get("right_knee_deg"),
            "top_framing_tag": dom.get("framing", {}).get("tag"),
            "top_framing_score": dom.get("framing", {}).get("score"),
            "camera_angle_zone": cam.get("zone"),
            "from_above_score": cam.get("from_above_score"),
            "from_below_score": cam.get("from_below_score"),
            "camera_pitch_proxy": cam.get("camera_pitch_proxy"),
            "feet_out_of_frame_score": cam.get("feet_out_of_frame_score"),
            "looking_back_score": round(float(gen.get("looking_back", 0.0)), 4),
            "from_behind_score": round(float(gen.get("from_behind", 0.0)), 4),
            "sitting_score": round(float(gen.get("sitting", 0.0)), 4),
            "squatting_score": round(float(gen.get("squatting", 0.0)), 4),
        }
        metric_rows.append(m_row)
        dino_list.append(emb)

        pose_vec = [
            float(m_row[c]) if (m_row[c] is not None and not math.isnan(float(m_row[c]))) else np.nan
            for c in pose_num_cols
        ]
        wd14_vec = [float(gen.get(t, 0.0)) for t in wd14_aux_tags]
        aux_list.append(pose_vec + wd14_vec)

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(f"[Extract {idx + 1}/{total}] latest={row['rel_path']}")

    dino_embs = np.vstack(dino_list).astype(np.float32)
    aux_feats = np.asarray(aux_list, dtype=np.float32)
    feat_df = pd.concat([dataset_df.reset_index(drop=True), pd.DataFrame(metric_rows)], axis=1)

    feat_df.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        cache_npz,
        dino_embs=dino_embs,
        aux_feats=aux_feats,
        aux_names=np.array(aux_names, dtype=object),
    )
    print(f"✅ 特徴量キャッシュを保存しました -> {cache_npz}")
    return feat_df, dino_embs, aux_feats, aux_names


def _run_oof_axis_audit(
    X_all: np.ndarray,
    labels: pd.Series,
    valid_mask: np.ndarray,
    n_splits: int = 5,
    min_class_count: int = 6,
    c_reg: float = 1.0,
) -> dict:
    """
    指定軸について Stratified K-Fold Out-of-Fold 予測を実行し、
    各画像の OOF 予測ラベル・信頼度・自クラス確率・混同行列を返す。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(labels)
    oof_pred = np.array([None] * n, dtype=object)
    oof_pred_prob = np.full(n, np.nan, dtype=np.float32)
    oof_label_prob = np.full(n, np.nan, dtype=np.float32)

    valid_labels = labels[valid_mask]
    counts = valid_labels.value_counts()
    keep_classes = sorted(counts[counts >= min_class_count].index.tolist())
    if len(keep_classes) < 2:
        return {
            "skipped": True,
            "reason": f"有効クラス数が 2 未満 (classes={keep_classes})",
            "oof_pred": oof_pred,
            "oof_pred_prob": oof_pred_prob,
            "oof_label_prob": oof_label_prob,
        }

    eval_mask = valid_mask & labels.isin(keep_classes).values
    eval_indices = np.where(eval_mask)[0]
    X_sub = X_all[eval_indices]
    y_sub = labels.iloc[eval_indices].values

    actual_splits = min(n_splits, int(counts[keep_classes].min()))
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    classes_arr = np.array(keep_classes)
    class_to_col = {c: i for i, c in enumerate(keep_classes)}

    proba_sub = np.zeros((len(eval_indices), len(keep_classes)), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X_sub, y_sub):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_reg,
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipe.fit(X_sub[tr_idx], y_sub[tr_idx])
        fold_probs = pipe.predict_proba(X_sub[va_idx])
        fold_classes = pipe.classes_
        for j, cls_name in enumerate(fold_classes):
            proba_sub[va_idx, class_to_col[cls_name]] = fold_probs[:, j]

    pred_cols = np.argmax(proba_sub, axis=1)
    pred_labels = classes_arr[pred_cols]
    pred_probs = proba_sub[np.arange(len(eval_indices)), pred_cols]
    true_probs = np.array(
        [proba_sub[i, class_to_col[y_sub[i]]] for i in range(len(eval_indices))],
        dtype=np.float32,
    )

    for loc_i, glob_i in enumerate(eval_indices):
        oof_pred[glob_i] = pred_labels[loc_i]
        oof_pred_prob[glob_i] = pred_probs[loc_i]
        oof_label_prob[glob_i] = true_probs[loc_i]

    acc = float(accuracy_score(y_sub, pred_labels))
    macro_f1 = float(f1_score(y_sub, pred_labels, average="macro"))
    cm = pd.DataFrame(
        confusion_matrix(y_sub, pred_labels, labels=keep_classes),
        index=[f"true:{c}" for c in keep_classes],
        columns=[f"pred:{c}" for c in keep_classes],
    )
    return {
        "skipped": False,
        "n_samples": int(len(eval_indices)),
        "n_splits": int(actual_splits),
        "classes": keep_classes,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "oof_pred": oof_pred,
        "oof_pred_prob": oof_pred_prob,
        "oof_label_prob": oof_label_prob,
    }


def run_dataset_audit(
    feat_df: pd.DataFrame,
    dino_embs: np.ndarray,
    aux_feats: np.ndarray,
    n_splits: int = 5,
    suspect_prob_thr: float = 0.70,
    outlier_z_thr: float = 2.5,
    cross_leaf_dup_sim: float = 0.985,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    3 階層フィルタリング済みデータセットに対して以下を一括監査する:
    1. 4 軸（orientation / framing / camera_angle / posture）の 5-fold OOF Classifier による入れ間違い候補検出
    2. 葉フォルダ内 DINOv2 外れ値 & 別フォルダ間ニア重複（移動漏れ）検出
    3. 物理・骨格の明らかな食い違いフラグ
    """
    df = feat_df.copy().reset_index(drop=True)
    X_all = np.hstack([dino_embs, aux_feats]).astype(np.float32)

    # 1. 各軸の 5-fold OOF Classifier 監査
    not_other = df["label_orientation"].values != "__other"
    not_special_fr = ~df["label_framing"].isin(SPECIAL_FRAMING_LEAVES | {"none"}).values

    axis_specs = [
        ("orientation", "label_orientation", not_other),
        ("framing", "label_framing", not_other & (df["label_framing"].values != "none")),
        (
            "camera_angle",
            "label_camera_angle",
            not_other & not_special_fr & (df["label_camera_angle"].values != "none"),
        ),
        (
            "posture",
            "label_posture",
            not_other & not_special_fr & (df["label_camera_angle"].values != "none"),
        ),
    ]

    axis_reports = {}
    for axis_name, col, mask in axis_specs:
        rep = _run_oof_axis_audit(X_all, df[col], mask, n_splits=n_splits)
        axis_reports[axis_name] = rep
        df[f"oof_pred_{axis_name}"] = rep["oof_pred"]
        df[f"oof_prob_{axis_name}"] = np.round(rep["oof_pred_prob"], 4)
        df[f"oof_true_prob_{axis_name}"] = np.round(rep["oof_label_prob"], 4)
        if not rep["skipped"]:
            print(
                f"[OOF {axis_name:12s}] n={rep['n_samples']:4d} | "
                f"acc={rep['accuracy']:.4f} | macro_f1={rep['macro_f1']:.4f} | "
                f"classes={rep['classes']}"
            )
        else:
            print(f"[OOF {axis_name:12s}] skipped: {rep['reason']}")

    # 2. DINOv2 葉内重心類似度（外れ値）& 全体最類似画像（別フォルダ間重複）
    sim_matrix = np.clip(dino_embs @ dino_embs.T, -1.0, 1.0)
    n = len(df)
    leaf_sims = np.full(n, np.nan, dtype=np.float32)
    leaf_sim_z = np.full(n, np.nan, dtype=np.float32)

    for leaf, grp in df.groupby("leaf_path"):
        idxs = grp.index.tolist()
        sub = dino_embs[idxs]
        c = sub.mean(axis=0)
        c /= max(float(np.linalg.norm(c)), 1e-8)
        sims = sub @ c
        leaf_sims[idxs] = sims
        if len(idxs) >= 5:
            m, sd = float(sims.mean()), max(float(sims.std(ddof=0)), 0.02)
            leaf_sim_z[idxs] = (sims - m) / sd

    df["leaf_centroid_sim"] = np.round(leaf_sims, 4)
    df["leaf_sim_z"] = np.round(leaf_sim_z, 2)

    sim_no_diag = sim_matrix.copy()
    np.fill_diagonal(sim_no_diag, -1.0)
    nearest_idx = np.argmax(sim_no_diag, axis=1)
    df["nearest_sim"] = np.round(sim_no_diag[np.arange(n), nearest_idx], 4) if n > 1 else 0.0
    df["nearest_rel_path"] = [df.loc[int(j), "rel_path"] for j in nearest_idx] if n > 1 else None
    df["nearest_leaf_path"] = [df.loc[int(j), "leaf_path"] for j in nearest_idx] if n > 1 else None

    # 3. 怪しい画像（Suspects）一覧を構築
    suspect_rows = []
    for i, row in df.iterrows():
        flags = []
        suggestions = {}

        # 3a. OOF 不一致（高信頼度）
        for axis_name, col, _ in axis_specs:
            pred = row[f"oof_pred_{axis_name}"]
            prob = row[f"oof_prob_{axis_name}"]
            true_p = row[f"oof_true_prob_{axis_name}"]
            if pred is not None and pred != row[col] and pd.notna(prob) and prob >= suspect_prob_thr:
                flags.append(f"oof_{axis_name}:{row[col]}->{pred}({prob:.2f},true={true_p:.2f})")
                suggestions[axis_name] = pred

        # 3b. 葉内 DINOv2 外れ値
        if pd.notna(row["leaf_sim_z"]) and row["leaf_sim_z"] <= -outlier_z_thr:
            flags.append(f"leaf_visual_outlier(sim={row['leaf_centroid_sim']:.3f},z={row['leaf_sim_z']:.2f})")

        # 3c. 別フォルダ間のほぼ同一画像 / 完全重複
        if row["exact_dup_count"] > 1:
            flags.append(f"exact_md5_dup(n={row['exact_dup_count']})")
        elif (
            row["nearest_sim"] >= cross_leaf_dup_sim
            and row["nearest_leaf_path"] != row["leaf_path"]
        ):
            flags.append(f"cross_leaf_dup({row['nearest_sim']:.3f}->{row['nearest_leaf_path']})")

        # 3d. 明らかな物理・骨格矛盾
        if row["label_orientation"] == "front" and row["body_facing"] == "back":
            flags.append("rule_conflict:front_but_body_back")
        if str(row["label_orientation"]).startswith("looking_back") and row["body_facing"] == "front" and float(row["from_behind_score"] or 0.0) < 0.25:
            flags.append("rule_conflict:looking_back_but_body_front")

        if flags:
            sug_ori = suggestions.get("orientation", row["label_orientation"])
            sug_fr = suggestions.get("framing", row["label_framing"])
            sug_ang = suggestions.get("camera_angle", row["label_camera_angle"])
            sug_pos = suggestions.get("posture", row["label_posture"])

            if sug_ori == "__other":
                suggested_leaf = "__other"
            elif sug_fr in SPECIAL_FRAMING_LEAVES or sug_fr == "none":
                suggested_leaf = f"{sug_ori}/{sug_fr}" if sug_fr != "none" else sug_ori
            else:
                ap = f"{sug_ang}__{sug_pos}" if sug_pos != "normal" else sug_ang
                suggested_leaf = f"{sug_ori}/{sug_fr}/{ap}" if sug_ang != "none" else f"{sug_ori}/{sug_fr}"

            # 優先度スコア（OOF 予測確率の最大値、なければ外れ値度合い）
            oof_probs = [
                float(row[f"oof_prob_{ax}"])
                for ax in suggestions.keys()
                if pd.notna(row[f"oof_prob_{ax}"])
            ]
            priority = max(oof_probs) if oof_probs else min(0.69, abs(float(row["leaf_sim_z"] or 0.0)) / 5.0)

            suspect_rows.append(
                {
                    "file_name": row["file_name"],
                    "file_path": row["file_path"],
                    "current_leaf": row["leaf_path"],
                    "suggested_leaf": suggested_leaf,
                    "priority": round(priority, 4),
                    "suspect_flags": "; ".join(flags),
                    "leaf_centroid_sim": row["leaf_centroid_sim"],
                    "body_facing": row["body_facing"],
                    "from_above_score": row["from_above_score"],
                    "from_behind_score": row["from_behind_score"],
                    "approve_move": False,
                }
            )

    suspects_df = (
        pd.DataFrame(suspect_rows)
        .sort_values("priority", ascending=False)
        .reset_index(drop=True)
        if suspect_rows
        else pd.DataFrame(
            columns=[
                "file_name",
                "file_path",
                "current_leaf",
                "suggested_leaf",
                "priority",
                "suspect_flags",
                "approve_move",
            ]
        )
    )
    print(f"\n🔍 監査完了: 怪しい候補 {len(suspects_df)} / {len(df)} 枚 ({len(suspects_df) / max(len(df), 1) * 100:.1f}%)")
    return df, suspects_df, axis_reports


def show_audit_suspects(
    suspects_df: pd.DataFrame,
    filter_keyword: str | None = None,
    n: int = 12,
    cols: int = 4,
):
    """
    怪しい候補（入れ間違い・外れ値）をサムネイル一覧で表示する。
    filter_keyword 例: "oof_orientation", "oof_framing", "oof_camera_angle", "oof_posture", "leaf_visual_outlier"
    """
    sub = suspects_df.copy()
    if filter_keyword:
        sub = sub[sub["suspect_flags"].str.contains(filter_keyword, na=False)]
    sub = sub.head(n)

    if len(sub) == 0:
        print(f"ℹ️ 条件に該当する怪しい候補はありません (filter={filter_keyword})。")
        return

    rows_n = math.ceil(len(sub) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.0, rows_n * 4.5))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Audit Suspects ({filter_keyword or 'ALL'} | showing {len(sub)})", fontsize=12)
    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sub.iterrows()):
        img = cv2.imread(row["file_path"])
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lines = [
            f"{row['file_name']} (p={row['priority']:.2f})",
            f"NOW: {row['current_leaf']}",
            f"SUG: {row['suggested_leaf']}",
            f"[{str(row['suspect_flags'])[:46]}]",
        ]
        ax.set_title("\n".join(lines), fontsize=7.8, color="#B71C1C")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def apply_relabel_moves(
    suspects_source,
    dataset_root: str,
    dry_run: bool = True,
) -> int:
    """
    `approve_move == True` にした行（CSV パスまたは DataFrame）を `suggested_leaf` フォルダへ安全に移動する。
    """
    df = pd.read_csv(suspects_source) if isinstance(suspects_source, (str, Path)) else suspects_source.copy()
    approved = df[df["approve_move"].astype(str).str.lower().isin(["true", "1"])]
    if len(approved) == 0:
        print("ℹ️ approve_move=True の行が 0 件のため移動なし。")
        return 0

    moved = 0
    for _, row in approved.iterrows():
        src = Path(row["file_path"])
        dst = Path(dataset_root) / str(row["suggested_leaf"]) / src.name
        if not src.exists():
            print(f"⚠️ スキップ（元ファイルなし）: {src}")
            continue
        if src.resolve() == dst.resolve():
            continue
        if dry_run:
            print(f"[DRY RUN] {src} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"✅ 移動: {row['current_leaf']} -> {row['suggested_leaf']} ({src.name})")
        moved += 1

    label = "[DRY RUN] 移動予定" if dry_run else "✅ 移動完了"
    print(f"{label}: {moved} 件")
    return moved
