"""Supervised scene classifier (flat base-scene + independent sitting/twisted modifiers)
using frozen DINOv2 embeddings and WD14-Tagger tag scores (no rtmlib required).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

EXCLUDED_SCENE_FRAMINGS: set[str] = {"broken_anatomy", "lower_body", "mid_body"}

DEFAULT_SCENE_ROUTING_CONFIG: dict = {
    "min_confidence": 0.45,
    "min_margin": 0.05,
    "sitting_threshold": 0.50,
    "twisted_threshold": 0.50,
    # True の場合、sitting と twisted が両方しきい値を超えたらスコアが高い方のみ採用する
    "exclusive_modifier": True,
}

DEFAULT_SCENE_FILTER_RULES: dict = {
    # WD14 タグベースの安全ネット（推論時の追加計算コストはゼロ）
    "enable_tag_filter": False,
    "required_tags": {
        "1girl": 0.70,
    },
    "forbidden_tags": {
        "monochrome": 0.50,
        "multiple_girls": 0.50,
    },
    # DINOv2 コサイン類似度ベースの葉内重複除外
    "enable_dedup": False,
    "max_leaf_pair_sim_dedup": 0.96,
}


def prepare_scene_dataset(
    dataset_df: pd.DataFrame,
    excluded_framings: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    `discover_labeled_dataset()` の出力からシーン分類用データセットを作成する。
    `broken_anatomy` / `lower_body` / `mid_body` などを除外し、
    フラットな `base_scene` と独立 2 値フラグ (`is_sitting`, `is_twisted`) を付与する。

    Returns
    -------
    train_df : pd.DataFrame
        学習対象の DataFrame（`base_scene`, `is_sitting`, `is_twisted` 列付き）
    excluded_df : pd.DataFrame
        除外された特殊クラスの DataFrame
    """
    excl = EXCLUDED_SCENE_FRAMINGS if excluded_framings is None else set(excluded_framings)
    is_excluded = dataset_df["label_framing"].isin(excl) | (
        dataset_df["label_camera_angle"] == "none"
    )
    excluded_df = dataset_df[is_excluded].copy().reset_index(drop=True)
    train_df = dataset_df[~is_excluded].copy().reset_index(drop=True)

    train_df["base_scene"] = (
        train_df["label_orientation"].astype(str)
        + "/"
        + train_df["label_framing"].astype(str)
        + "/"
        + train_df["label_camera_angle"].astype(str)
    )
    train_df["is_sitting"] = (train_df["label_posture"] == "sitting").astype(int)
    train_df["is_twisted"] = (train_df["label_posture"] == "twisted").astype(int)
    return train_df, excluded_df


def _compute_file_md5(file_path: str) -> str:
    return hashlib.md5(Path(file_path).read_bytes()).hexdigest()


def extract_or_load_scene_features(
    items_df: pd.DataFrame,
    cache_dir: str,
    embedder,
    tagger=None,
    force_recompute: bool = False,
    cache_prefix: str = "scene_features",
    progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """
    DINOv2 埋め込みと WD14-Tagger の全 general タグスコア（float16）を抽出し、
    md5 キーで `cache_dir` にキャッシュする（rtmlib 不使用）。

    Returns
    -------
    dino_embs : np.ndarray (N, D) float32
    wd14_probs : np.ndarray (N, G) float32 or None
    wd14_tag_names : list[str]
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_npz = os.path.join(cache_dir, f"{cache_prefix}.npz")
    cache_meta = os.path.join(cache_dir, f"{cache_prefix}_meta.json")

    if "md5" in items_df.columns:
        md5_list = items_df["md5"].astype(str).tolist()
    else:
        md5_list = [_compute_file_md5(p) for p in items_df["file_path"].tolist()]

    want_wd14 = tagger is not None

    if not force_recompute and os.path.exists(cache_npz) and os.path.exists(cache_meta):
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached_md5s = meta.get("md5_list", [])
        has_wd14 = bool(meta.get("has_wd14", False))
        if set(md5_list).issubset(set(cached_md5s)) and (not want_wd14 or has_wd14):
            data = np.load(cache_npz, allow_pickle=False)
            idx_map = {m: i for i, m in enumerate(cached_md5s)}
            order = [idx_map[m] for m in md5_list]
            dino_embs = data["dino_embs"][order].astype(np.float32)
            if want_wd14 and "wd14_probs" in data:
                wd14_probs = data["wd14_probs"][order].astype(np.float32)
                wd14_tag_names = list(meta.get("wd14_tag_names", []))
            else:
                wd14_probs = None
                wd14_tag_names = []
            print(
                f"✅ キャッシュから特徴量を読み込みました ({len(md5_list)} 枚 | "
                f"DINOv2={dino_embs.shape}, WD14={None if wd14_probs is None else wd14_probs.shape})"
            )
            return dino_embs, wd14_probs, wd14_tag_names

    # 既存キャッシュがあればヒット分だけ再利用し、未計算画像のみ推論する
    existing_dino: dict[str, np.ndarray] = {}
    existing_wd14: dict[str, np.ndarray] = {}
    cached_tag_names: list[str] = []
    if not force_recompute and os.path.exists(cache_npz) and os.path.exists(cache_meta):
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached_md5s = meta.get("md5_list", [])
        cached_tag_names = list(meta.get("wd14_tag_names", []))
        data = np.load(cache_npz, allow_pickle=False)
        for i, m in enumerate(cached_md5s):
            existing_dino[m] = data["dino_embs"][i]
            if "wd14_probs" in data:
                existing_wd14[m] = data["wd14_probs"][i]

    wd14_tag_names: list[str] = (
        [tagger.tag_names[i] for i in tagger.general_indices] if want_wd14 else []
    )
    can_reuse_wd14 = want_wd14 and (cached_tag_names == wd14_tag_names)

    dino_rows: list[np.ndarray] = []
    wd14_rows: list[np.ndarray] = []
    n_total = len(items_df)
    n_reused = 0

    for idx, (file_path, md5) in enumerate(
        zip(items_df["file_path"].tolist(), md5_list), start=1
    ):
        need_dino = md5 not in existing_dino
        need_wd14 = want_wd14 and (not can_reuse_wd14 or md5 not in existing_wd14)

        if not need_dino and not need_wd14:
            dino_rows.append(existing_dino[md5].astype(np.float32))
            if want_wd14:
                wd14_rows.append(existing_wd14[md5].astype(np.float16))
            n_reused += 1
        else:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise RuntimeError(f"画像の読み込みに失敗しました: {file_path}")
            if need_dino:
                emb = embedder.embed(img_bgr).astype(np.float32)
            else:
                emb = existing_dino[md5].astype(np.float32)
            dino_rows.append(emb)

            if want_wd14:
                if need_wd14:
                    pred = tagger.predict(img_bgr)["general"]
                    vec = np.array(
                        [pred[t] for t in wd14_tag_names], dtype=np.float16
                    )
                else:
                    vec = existing_wd14[md5].astype(np.float16)
                wd14_rows.append(vec)

        if progress_every and (idx % progress_every == 0 or idx == n_total):
            print(f"  特徴抽出進捗: {idx}/{n_total} 枚 (キャッシュ再利用: {n_reused} 枚)")

    dino_embs = np.vstack(dino_rows).astype(np.float32)
    save_dict = {"dino_embs": dino_embs.astype(np.float32)}
    if want_wd14:
        wd14_f16 = np.vstack(wd14_rows).astype(np.float16)
        save_dict["wd14_probs"] = wd14_f16
        wd14_probs = wd14_f16.astype(np.float32)
    else:
        wd14_probs = None

    np.savez_compressed(cache_npz, **save_dict)
    meta_out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(md5_list),
        "md5_list": md5_list,
        "has_wd14": want_wd14,
        "wd14_tag_names": wd14_tag_names,
        "dinov2_model_id": getattr(embedder, "model_id", None),
        "wd14_repo_id": getattr(tagger, "repo_id", None) if want_wd14 else None,
    }
    with open(cache_meta, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(
        f"💾 特徴量をキャッシュ保存しました: {cache_npz} "
        f"(DINOv2={dino_embs.shape}, WD14={None if wd14_probs is None else wd14_probs.shape})"
    )
    return dino_embs, wd14_probs, wd14_tag_names


def select_wd14_active_tags(
    wd14_probs: np.ndarray,
    wd14_tag_names: list[str],
    min_max_score: float = 0.15,
    max_tags: int = 512,
) -> tuple[list[int], list[str]]:
    """
    学習データ内で有意に発火する WD14 general タグの上位を選択する。
    """
    max_vals = wd14_probs.max(axis=0)
    stds = wd14_probs.std(axis=0)
    candidates = np.where((max_vals >= min_max_score) & (stds > 1e-4))[0]
    if len(candidates) == 0:
        candidates = np.argsort(-stds)[: min(max_tags, wd14_probs.shape[1])]
    else:
        order = np.argsort(-stds[candidates])
        candidates = candidates[order[:max_tags]]
    selected_idx = [int(i) for i in sorted(candidates.tolist())]
    selected_names = [wd14_tag_names[i] for i in selected_idx]
    return selected_idx, selected_names


def combine_scene_features(
    dino_embs: np.ndarray,
    wd14_probs: np.ndarray | None = None,
    selected_wd14_indices: list[int] | None = None,
    feature_mode: str = "dino_wd14",
    wd14_weight: float = 1.0,
) -> np.ndarray:
    """
    DINOv2 埋め込み（L2 正規化済み）と選択済み WD14 タグベクトル（L2 正規化済み）を結合する。
    """
    dino_norm = dino_embs.astype(np.float32)
    norms = np.linalg.norm(dino_norm, axis=1, keepdims=True)
    dino_norm = dino_norm / np.clip(norms, 1e-8, None)

    if feature_mode == "dino_only" or wd14_probs is None:
        return dino_norm

    if selected_wd14_indices is not None:
        sub = wd14_probs[:, selected_wd14_indices].astype(np.float32)
    else:
        sub = wd14_probs.astype(np.float32)

    sub_norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sub_norm = sub / np.clip(sub_norms, 1e-8, None)
    return np.hstack([dino_norm, sub_norm * float(wd14_weight)]).astype(np.float32)


class EncodedClassifier:
    """
    scikit-learn 1.6+ の MLPClassifier 等で文字列ラベルが `np.isnan` TypeError を起こすのを防ぐため、
    内部で LabelEncoder による整数ラベルへ変換して学習・推論する軽量ラッパー（joblib 保存対応）。
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.label_encoder = LabelEncoder()
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_enc = self.label_encoder.fit_transform(y)
        self.estimator.fit(X, y_enc)
        self.classes_ = self.label_encoder.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_enc = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(y_enc)


class SafeLinearSVC:
    """
    最小クラス件数が 3 未満（例: 1 枚クラスあり）の場合でも落ちない LinearSVC 確率出力ラッパー。
    min_count >= 2 なら CalibratedClassifierCV(cv=min(3, min_count))、
    min_count < 2 なら LinearSVC の decision_function に softmax/sigmoid を適用する。
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight=None,
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.use_calibrated = False
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        _, counts = np.unique(y, return_counts=True)
        min_cnt = int(counts.min()) if len(counts) > 0 else 0
        base = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        if min_cnt >= 2:
            self.use_calibrated = True
            self.model = CalibratedClassifierCV(base, cv=min(3, min_cnt))
        else:
            self.use_calibrated = False
            self.model = base
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_calibrated:
            return self.model.predict_proba(X)
        scores = self.model.decision_function(X)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exps = np.exp(np.clip(shifted, -30.0, 0.0))
        return (exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-8, None)).astype(np.float32)


def _build_estimator(model_name: str, random_state: int = 42):
    if model_name == "logreg_l2":
        return LogisticRegression(
            C=5.0,
            max_iter=2000,
            random_state=random_state,
        )
    if model_name == "logreg_balanced":
        return LogisticRegression(
            C=5.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    if model_name == "logreg_strong":
        return LogisticRegression(
            C=15.0,
            max_iter=2000,
            random_state=random_state,
        )
    if model_name == "logreg_c30":
        return LogisticRegression(
            C=30.0,
            max_iter=2000,
            random_state=random_state,
        )
    if model_name == "mlp":
        return EncodedClassifier(
            MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=1e-2,
                batch_size=128,
                max_iter=100,
                tol=1e-3,
                n_iter_no_change=5,
                early_stopping=False,
                random_state=random_state,
            )
        )
    if model_name == "linear_svc_cal":
        return SafeLinearSVC(C=1.0, max_iter=5000, random_state=random_state)
    raise ValueError(f"Unknown model_name: {model_name}")


def _build_binary_estimator(model_name: str, random_state: int = 42):
    if model_name == "mlp":
        return EncodedClassifier(
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=1e-2,
                batch_size=128,
                max_iter=100,
                tol=1e-3,
                n_iter_no_change=5,
                early_stopping=False,
                random_state=random_state,
            )
        )
    if model_name == "linear_svc_cal":
        return SafeLinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=random_state
        )
    return LogisticRegression(
        C=3.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )


def evaluate_scene_classifiers(
    train_df: pd.DataFrame,
    dino_embs: np.ndarray,
    wd14_probs: np.ndarray | None,
    wd14_tag_names: list[str],
    feature_modes: list[str] | None = None,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    wd14_min_max_score: float = 0.15,
    wd14_max_tags: int = 512,
    wd14_weight: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    複数の特徴量セット (`dino_only`, `dino_wd14`) × 分類器候補を Stratified K-Fold CV で比較評価する。
    サンプル数が `n_splits` 未満の極小クラスは全 fold の train に含め、評価は `n >= n_splits` クラスで算出する。

    Returns
    -------
    comparison_df : pd.DataFrame
        モデル比較結果テーブル（`macro_f1` 降順）
    best_oof : dict
        最良モデルの OOF 予測情報（`pred_df`, `config`, `confusion_pairs` 等）
    """
    if feature_modes is None:
        feature_modes = ["dino_only", "dino_wd14"] if wd14_probs is not None else ["dino_only"]
    if model_names is None:
        model_names = ["logreg_l2", "logreg_balanced", "logreg_strong", "linear_svc_cal"]

    y_base = train_df["base_scene"].astype(str).to_numpy()
    y_sit = train_df["is_sitting"].astype(int).to_numpy()
    y_twi = train_df["is_twisted"].astype(int).to_numpy()

    counts = pd.Series(y_base).value_counts()
    eval_mask = np.array([counts[c] >= n_splits for c in y_base])
    eval_indices = np.where(eval_mask)[0]
    rare_indices = np.where(~eval_mask)[0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(eval_indices, y_base[eval_indices]))

    # sitting / twisted は全件に対して Stratified K-Fold
    skf_sit = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_sit = list(skf_sit.split(np.arange(len(y_sit)), y_sit))
    skf_twi = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_twi = list(skf_twi.split(np.arange(len(y_twi)), y_twi))

    rows = []
    best_score = -1.0
    best_oof: dict = {}

    def _build_fold_matrices(fold_pairs, fmode: str, is_base: bool):
        mats = []
        for a, b in fold_pairs:
            if is_base:
                tr_idx = np.concatenate([eval_indices[a], rare_indices])
                va_idx = eval_indices[b]
                va_target = b
            else:
                tr_idx, va_idx = a, b
                va_target = b

            if fmode == "dino_wd14" and wd14_probs is not None:
                sel_idx, _ = select_wd14_active_tags(
                    wd14_probs[tr_idx],
                    wd14_tag_names,
                    min_max_score=wd14_min_max_score,
                    max_tags=wd14_max_tags,
                )
            else:
                sel_idx = None

            X_tr = combine_scene_features(
                dino_embs[tr_idx],
                wd14_probs[tr_idx] if wd14_probs is not None else None,
                selected_wd14_indices=sel_idx,
                feature_mode=fmode,
                wd14_weight=wd14_weight,
            )
            X_va = combine_scene_features(
                dino_embs[va_idx],
                wd14_probs[va_idx] if wd14_probs is not None else None,
                selected_wd14_indices=sel_idx,
                feature_mode=fmode,
                wd14_weight=wd14_weight,
            )
            mats.append((tr_idx, va_target, X_tr, X_va))
        return mats

    for fmode in feature_modes:
        if fmode == "dino_wd14" and wd14_probs is None:
            continue

        # 特徴量行列を fold ごとに事前計算（モデル間で再利用）
        base_fold_mats = _build_fold_matrices(folds, fmode, is_base=True)
        sit_fold_mats = _build_fold_matrices(folds_sit, fmode, is_base=False)
        twi_fold_mats = _build_fold_matrices(folds_twi, fmode, is_base=False)
        binary_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for mname in model_names:
            t0 = time.perf_counter()
            oof_pred = np.empty(len(eval_indices), dtype=object)
            oof_top1_prob = np.zeros(len(eval_indices), dtype=np.float32)
            oof_top2_label = np.empty(len(eval_indices), dtype=object)
            oof_top2_prob = np.zeros(len(eval_indices), dtype=np.float32)
            eval_classes = sorted(np.unique(y_base[eval_indices]).tolist())
            class_to_col = {c: i for i, c in enumerate(eval_classes)}
            oof_prob_mat = np.zeros((len(eval_indices), len(eval_classes)), dtype=np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr_idx, va_rel, X_tr, X_va in base_fold_mats:
                    clf = _build_estimator(mname, random_state=random_state)
                    clf.fit(X_tr, y_base[tr_idx])
                    probs = clf.predict_proba(X_va)
                    clf_classes = clf.classes_

                    order = np.argsort(-probs, axis=1)
                    top1_idx = order[:, 0]
                    top2_idx = order[:, 1] if probs.shape[1] > 1 else order[:, 0]

                    oof_pred[va_rel] = clf_classes[top1_idx]
                    oof_top1_prob[va_rel] = probs[np.arange(len(va_rel)), top1_idx]
                    oof_top2_label[va_rel] = clf_classes[top2_idx]
                    oof_top2_prob[va_rel] = probs[np.arange(len(va_rel)), top2_idx]

                    for c_pos, c_name in enumerate(clf_classes):
                        if c_name in class_to_col:
                            oof_prob_mat[va_rel, class_to_col[c_name]] = probs[:, c_pos]

                # sitting / twisted の OOF 評価（同一2値推定器はキャッシュ再利用）
                bin_key = mname if mname in ("mlp", "linear_svc_cal") else "logreg"
                if bin_key in binary_cache:
                    oof_sit_prob, oof_twi_prob = binary_cache[bin_key]
                else:
                    oof_sit_prob = np.zeros(len(y_sit), dtype=np.float32)
                    for tr_idx, va_idx, X_tr, X_va in sit_fold_mats:
                        clf_sit = _build_binary_estimator(mname, random_state=random_state)
                        clf_sit.fit(X_tr, y_sit[tr_idx])
                        oof_sit_prob[va_idx] = clf_sit.predict_proba(X_va)[:, 1]

                    oof_twi_prob = np.zeros(len(y_twi), dtype=np.float32)
                    for tr_idx, va_idx, X_tr, X_va in twi_fold_mats:
                        clf_twi = _build_binary_estimator(mname, random_state=random_state)
                        clf_twi.fit(X_tr, y_twi[tr_idx])
                        oof_twi_prob[va_idx] = clf_twi.predict_proba(X_va)[:, 1]
                    binary_cache[bin_key] = (oof_sit_prob, oof_twi_prob)

            elapsed = time.perf_counter() - t0
            y_true_eval = y_base[eval_indices]
            acc = float(accuracy_score(y_true_eval, oof_pred))
            macro_f1 = float(
                f1_score(y_true_eval, oof_pred, average="macro", zero_division=0)
            )
            weighted_f1 = float(
                f1_score(y_true_eval, oof_pred, average="weighted", zero_division=0)
            )
            # top-3 accuracy
            row_sums = oof_prob_mat.sum(axis=1, keepdims=True)
            norm_prob_mat = oof_prob_mat / np.clip(row_sums, 1e-8, None)
            y_true_col = np.array([class_to_col[c] for c in y_true_eval])
            top3 = float(
                top_k_accuracy_score(
                    y_true_col,
                    norm_prob_mat,
                    k=min(3, len(eval_classes)),
                    labels=np.arange(len(eval_classes)),
                )
            )

            sit_f1 = float(f1_score(y_sit, (oof_sit_prob >= 0.5).astype(int), zero_division=0))
            sit_ap = float(average_precision_score(y_sit, oof_sit_prob)) if y_sit.sum() > 0 else 0.0
            twi_f1 = float(f1_score(y_twi, (oof_twi_prob >= 0.5).astype(int), zero_division=0))
            twi_ap = float(average_precision_score(y_twi, oof_twi_prob)) if y_twi.sum() > 0 else 0.0

            row = {
                "feature_mode": fmode,
                "model_name": mname,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "top3_accuracy": round(top3, 4),
                "sitting_f1": round(sit_f1, 4),
                "sitting_ap": round(sit_ap, 4),
                "twisted_f1": round(twi_f1, 4),
                "twisted_ap": round(twi_ap, 4),
                "elapsed_sec": round(elapsed, 1),
                "eval_samples": int(len(eval_indices)),
                "eval_classes": int(len(eval_classes)),
                "rare_train_only_classes": int(len(counts) - len(eval_classes)),
            }
            rows.append(row)
            print(
                f"[{fmode:9s} | {mname:15s}] ({elapsed:5.1f}s) "
                f"acc={acc:.4f}, macro_f1={macro_f1:.4f}, top3={top3:.4f} | "
                f"sit_f1={sit_f1:.4f} (AP={sit_ap:.4f}), twi_f1={twi_f1:.4f} (AP={twi_ap:.4f})"
            )

            # 選定基準: macro_f1 と accuracy の平均
            composite = 0.5 * macro_f1 + 0.5 * acc
            if composite > best_score:
                best_score = composite
                oof_df = train_df.iloc[eval_indices].copy().reset_index(drop=True)
                oof_df["oof_pred"] = oof_pred
                oof_df["oof_prob"] = oof_top1_prob
                oof_df["oof_top2_label"] = oof_top2_label
                oof_df["oof_top2_prob"] = oof_top2_prob
                oof_df["oof_margin"] = oof_top1_prob - oof_top2_prob
                oof_df["oof_correct"] = oof_df["base_scene"] == oof_df["oof_pred"]
                oof_df["oof_sitting_prob"] = oof_sit_prob[eval_indices]
                oof_df["oof_twisted_prob"] = oof_twi_prob[eval_indices]

                wrong = oof_df[~oof_df["oof_correct"]]
                conf_pairs = (
                    wrong.groupby(["base_scene", "oof_pred"])
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                    .reset_index(drop=True)
                )
                best_oof = {
                    "config": {
                        "feature_mode": fmode,
                        "model_name": mname,
                        "wd14_min_max_score": wd14_min_max_score,
                        "wd14_max_tags": wd14_max_tags,
                        "wd14_weight": wd14_weight,
                    },
                    "metrics": row,
                    "oof_df": oof_df,
                    "confusion_pairs": conf_pairs,
                }

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    return comparison_df, best_oof


def train_scene_classifier_bundle(
    train_df: pd.DataFrame,
    dino_embs: np.ndarray,
    wd14_probs: np.ndarray | None,
    wd14_tag_names: list[str],
    feature_mode: str = "dino_wd14",
    model_name: str = "logreg_l2",
    wd14_min_max_score: float = 0.15,
    wd14_max_tags: int = 512,
    wd14_weight: float = 1.0,
    dinov2_model_id: str | None = None,
    wd14_repo_id: str | None = None,
    cv_metrics: dict | None = None,
    random_state: int = 42,
) -> dict:
    """
    全学習データ（極小クラス含む全ベースシーン）で本番モデルバンドルを学習する。
    """
    if feature_mode == "dino_wd14" and wd14_probs is not None:
        selected_wd14_indices, selected_wd14_names = select_wd14_active_tags(
            wd14_probs,
            wd14_tag_names,
            min_max_score=wd14_min_max_score,
            max_tags=wd14_max_tags,
        )
    else:
        selected_wd14_indices = []
        selected_wd14_names = []

    X = combine_scene_features(
        dino_embs,
        wd14_probs,
        selected_wd14_indices=selected_wd14_indices if selected_wd14_indices else None,
        feature_mode=feature_mode,
        wd14_weight=wd14_weight,
    )
    y_base = train_df["base_scene"].astype(str).to_numpy()
    y_sit = train_df["is_sitting"].astype(int).to_numpy()
    y_twi = train_df["is_twisted"].astype(int).to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_clf = _build_estimator(model_name, random_state=random_state)
        base_clf.fit(X, y_base)

        sitting_clf = None
        if len(np.unique(y_sit)) >= 2:
            sitting_clf = _build_binary_estimator(model_name, random_state=random_state)
            sitting_clf.fit(X, y_sit)

        twisted_clf = None
        if len(np.unique(y_twi)) >= 2:
            twisted_clf = _build_binary_estimator(model_name, random_state=random_state)
            twisted_clf.fit(X, y_twi)

    # クラスごとの DINOv2 重心（推論時の異常度/類似度モニタリング用）
    class_centroids: dict[str, np.ndarray] = {}
    for cls in sorted(np.unique(y_base).tolist()):
        embs = dino_embs[y_base == cls]
        mean_vec = embs.mean(axis=0).astype(np.float32)
        mean_vec = mean_vec / max(float(np.linalg.norm(mean_vec)), 1e-8)
        class_centroids[cls] = mean_vec

    class_counts = {str(k): int(v) for k, v in pd.Series(y_base).value_counts().items()}

    bundle = {
        "version": "scene_classifier_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_mode": feature_mode,
        "model_name": model_name,
        "wd14_weight": float(wd14_weight),
        "wd14_min_max_score": float(wd14_min_max_score),
        "wd14_max_tags": int(wd14_max_tags),
        "selected_wd14_tag_names": selected_wd14_names,
        "dinov2_model_id": dinov2_model_id,
        "wd14_repo_id": wd14_repo_id,
        "base_classes": [str(c) for c in base_clf.classes_.tolist()],
        "class_counts": class_counts,
        "n_train_samples": int(len(train_df)),
        "n_sitting_pos": int(y_sit.sum()),
        "n_twisted_pos": int(y_twi.sum()),
        "cv_metrics": cv_metrics or {},
        "base_clf": base_clf,
        "sitting_clf": sitting_clf,
        "twisted_clf": twisted_clf,
        "class_centroids": class_centroids,
    }
    print(
        f"✅ 本番モデル学習完了: {len(bundle['base_classes'])} ベースクラス "
        f"({len(train_df)} 枚 | mode={feature_mode}, model={model_name}, "
        f"wd14_tags={len(selected_wd14_names)})"
    )
    return bundle


def save_scene_classifier_bundle(bundle: dict, model_dir: str) -> dict:
    """モデルバンドル (.joblib) と人間可読な metadata.json を保存する。"""
    os.makedirs(model_dir, exist_ok=True)
    bundle_path = os.path.join(model_dir, "scene_classifier_bundle.joblib")
    meta_path = os.path.join(model_dir, "scene_classifier_metadata.json")

    joblib.dump(bundle, bundle_path, compress=3)

    meta = {
        k: v
        for k, v in bundle.items()
        if k not in ("base_clf", "sitting_clf", "twisted_clf", "class_centroids")
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"💾 モデル保存完了: {bundle_path}")
    print(f"💾 メタデータ保存完了: {meta_path}")
    return {"bundle_path": bundle_path, "meta_path": meta_path}


def load_scene_classifier_bundle(model_dir: str) -> dict:
    """保存済みモデルバンドル (.joblib) を読み込む。"""
    bundle_path = os.path.join(model_dir, "scene_classifier_bundle.joblib")
    assert os.path.exists(bundle_path), f"モデルファイルが見つかりません: {bundle_path}"
    bundle = joblib.load(bundle_path)
    print(
        f"✅ モデル読み込み完了: {bundle_path} "
        f"({len(bundle['base_classes'])} classes, mode={bundle['feature_mode']}, "
        f"model={bundle['model_name']})"
    )
    return bundle


def _compose_leaf_name(
    base_scene: str,
    sitting_prob: float,
    twisted_prob: float,
    routing_config: dict,
) -> tuple[str, str]:
    sit_thr = float(routing_config.get("sitting_threshold", 0.50))
    twi_thr = float(routing_config.get("twisted_threshold", 0.50))
    exclusive = bool(routing_config.get("exclusive_modifier", True))

    is_sit = sitting_prob >= sit_thr
    is_twi = twisted_prob >= twi_thr

    if is_sit and is_twi:
        if exclusive:
            mod = "sitting" if sitting_prob >= twisted_prob else "twisted"
            return f"{base_scene}__{mod}", mod
        return f"{base_scene}__sitting__twisted", "sitting__twisted"
    if is_sit:
        return f"{base_scene}__sitting", "sitting"
    if is_twi:
        return f"{base_scene}__twisted", "twisted"
    return base_scene, "normal"


def predict_scene_batch(
    image_items: list[dict],
    bundle: dict,
    embedder,
    tagger=None,
    cache_dir: str | None = None,
    routing_config: dict | None = None,
    filter_rules: dict | None = None,
    force_recompute: bool = False,
    use_hflip_tta: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    未分類画像群に対し、学習済みシーン分類器（v1 / v2 自動判別）で
    推論・信頼度ルーティング・オプションフィルタを実行する。
    v2 モデルかつ `use_hflip_tta=True` の場合、水平フリップ TTA で左右対称に確率平均する。
    """
    rcfg = copy.deepcopy(DEFAULT_SCENE_ROUTING_CONFIG)
    if routing_config:
        rcfg.update(routing_config)

    fcfg = copy.deepcopy(DEFAULT_SCENE_FILTER_RULES)
    if filter_rules:
        fcfg.update(filter_rules)

    items_df = pd.DataFrame(image_items)
    assert len(items_df) > 0, "推論対象の画像が 0 件です。"

    is_v2 = str(bundle.get("version", "")).startswith("scene_classifier_v2")
    fmode = str(bundle["feature_mode"])
    need_wd14 = ("wd14" in fmode) or bool(fcfg.get("enable_tag_filter", False))
    if need_wd14 and tagger is None:
        raise ValueError("このモデル/フィルタ設定では WD14Tagger のインスタンスが必要です。")

    use_cache_dir = cache_dir or "./cache/scene_inference"
    selected_names = bundle.get("selected_wd14_tag_names", [])

    if is_v2:
        fpack = extract_or_load_scene_feature_pack(
            items_df=items_df,
            cache_dir=use_cache_dir,
            embedder=embedder,
            tagger=tagger if need_wd14 else None,
            include_hflip=use_hflip_tta,
            force_recompute=force_recompute,
            cache_prefix="infer_features",
        )
        dino_embs = fpack["dino_cls"]
        wd14_probs = fpack["wd14_probs"]
        wd14_tag_names = fpack["wd14_tag_names"]

        if "wd14" in fmode:
            assert wd14_probs is not None
            tag_to_idx = {name: i for i, name in enumerate(wd14_tag_names)}
            missing = [t for t in selected_names if t not in tag_to_idx]
            if missing:
                raise ValueError(
                    f"推論側 WD14Tagger に学習時のタグが見つかりません: {missing[:5]}"
                )
            selected_indices = [tag_to_idx[t] for t in selected_names]
        else:
            selected_indices = None

        spatial_weight = float(bundle.get("spatial_weight", 0.7))
        wd14_weight = float(bundle.get("wd14_weight", 1.0))

        X = combine_scene_features_v2(
            feature_pack=fpack,
            flipped=False,
            selected_wd14_indices=selected_indices,
            feature_mode=fmode,
            spatial_weight=spatial_weight,
            wd14_weight=wd14_weight,
        )
        base_clf = bundle["base_clf"]
        clf_classes = base_clf.classes_
        probs = base_clf.predict_proba(X)

        if bundle.get("sitting_clf") is not None:
            sit_probs = bundle["sitting_clf"].predict_proba(X)[:, 1]
        else:
            sit_probs = np.zeros(len(items_df), dtype=np.float32)

        if bundle.get("twisted_clf") is not None:
            twi_probs = bundle["twisted_clf"].predict_proba(X)[:, 1]
        else:
            twi_probs = np.zeros(len(items_df), dtype=np.float32)

        if use_hflip_tta and fpack.get("dino_cls_flip") is not None:
            X_flip = combine_scene_features_v2(
                feature_pack=fpack,
                flipped=True,
                selected_wd14_indices=selected_indices,
                feature_mode=fmode,
                spatial_weight=spatial_weight,
                wd14_weight=wd14_weight,
            )
            probs_flip = base_clf.predict_proba(X_flip)
            # 反転画像の予測確率を、元画像側の左右ラベル列へマップして平均する
            cls_to_col = {str(c): idx for idx, c in enumerate(clf_classes)}
            aligned_flip = np.zeros_like(probs_flip)
            for col_idx, cls_name in enumerate(clf_classes):
                unflipped_name = flip_scene_label(str(cls_name))
                dst_col = cls_to_col.get(unflipped_name)
                if dst_col is not None:
                    aligned_flip[:, dst_col] += probs_flip[:, col_idx]
                else:
                    aligned_flip[:, col_idx] += probs_flip[:, col_idx]
            probs = 0.5 * (probs + aligned_flip)

            if bundle.get("sitting_clf") is not None:
                sit_probs = 0.5 * (
                    sit_probs + bundle["sitting_clf"].predict_proba(X_flip)[:, 1]
                )
            if bundle.get("twisted_clf") is not None:
                twi_probs = 0.5 * (
                    twi_probs + bundle["twisted_clf"].predict_proba(X_flip)[:, 1]
                )
    else:
        dino_embs, wd14_probs, wd14_tag_names = extract_or_load_scene_features(
            items_df=items_df,
            cache_dir=use_cache_dir,
            embedder=embedder,
            tagger=tagger if need_wd14 else None,
            force_recompute=force_recompute,
            cache_prefix="infer_features",
        )

        if fmode == "dino_wd14":
            assert wd14_probs is not None
            tag_to_idx = {name: i for i, name in enumerate(wd14_tag_names)}
            missing = [t for t in selected_names if t not in tag_to_idx]
            if missing:
                raise ValueError(
                    f"推論側 WD14Tagger に学習時のタグが見つかりません: {missing[:5]}"
                )
            selected_indices = [tag_to_idx[t] for t in selected_names]
        else:
            selected_indices = None

        X = combine_scene_features(
            dino_embs=dino_embs,
            wd14_probs=wd14_probs,
            selected_wd14_indices=selected_indices,
            feature_mode=fmode,
            wd14_weight=bundle.get("wd14_weight", 1.0),
        )

        base_clf = bundle["base_clf"]
        probs = base_clf.predict_proba(X)
        clf_classes = base_clf.classes_

        if bundle.get("sitting_clf") is not None:
            sit_probs = bundle["sitting_clf"].predict_proba(X)[:, 1]
        else:
            sit_probs = np.zeros(len(items_df), dtype=np.float32)

        if bundle.get("twisted_clf") is not None:
            twi_probs = bundle["twisted_clf"].predict_proba(X)[:, 1]
        else:
            twi_probs = np.zeros(len(items_df), dtype=np.float32)

    order = np.argsort(-probs, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if probs.shape[1] > 1 else order[:, 0]

    top1_labels = clf_classes[top1_idx]
    top1_probs = probs[np.arange(len(items_df)), top1_idx]
    top2_labels = clf_classes[top2_idx]
    top2_probs = probs[np.arange(len(items_df)), top2_idx]
    margins = top1_probs - top2_probs

    centroids = bundle.get("class_centroids", {})
    tag_name_to_idx = {t: i for i, t in enumerate(wd14_tag_names)} if wd14_tag_names else {}

    rows = []
    min_conf = float(rcfg.get("min_confidence", 0.45))
    min_margin = float(rcfg.get("min_margin", 0.05))
    enable_tag_filter = bool(fcfg.get("enable_tag_filter", False))
    req_tags = fcfg.get("required_tags", {}) or {}
    forb_tags = fcfg.get("forbidden_tags", {}) or {}

    for i in range(len(items_df)):
        base_scene = str(top1_labels[i])
        conf = float(top1_probs[i])
        margin = float(margins[i])
        sp = float(sit_probs[i])
        tp = float(twi_probs[i])
        composed_leaf, modifier = _compose_leaf_name(base_scene, sp, tp, rcfg)

        centroid = centroids.get(base_scene)
        centroid_sim = float(np.dot(dino_embs[i], centroid)) if centroid is not None else math.nan

        reject_reasons = []
        if enable_tag_filter and wd14_probs is not None:
            for tname, thr in req_tags.items():
                if tname in tag_name_to_idx:
                    score = float(wd14_probs[i, tag_name_to_idx[tname]])
                    if score < float(thr):
                        reject_reasons.append(f"missing_{tname}({score:.2f}<{thr})")
            for tname, thr in forb_tags.items():
                if tname in tag_name_to_idx:
                    score = float(wd14_probs[i, tag_name_to_idx[tname]])
                    if score >= float(thr):
                        reject_reasons.append(f"forbidden_{tname}({score:.2f}>={thr})")

        if reject_reasons:
            route = "rejected"
            final_leaf = f"_rejected/{reject_reasons[0].split('(')[0]}"
            reason_str = "; ".join(reject_reasons)
        elif conf < min_conf or margin < min_margin:
            route = "uncertain"
            final_leaf = "_uncertain"
            u_reasons = []
            if conf < min_conf:
                u_reasons.append(f"low_conf({conf:.2f}<{min_conf})")
            if margin < min_margin:
                u_reasons.append(f"low_margin({margin:.2f}<{min_margin})")
            reason_str = "; ".join(u_reasons)
        else:
            route = "classified"
            final_leaf = composed_leaf
            reason_str = ""

        parts = base_scene.split("/")
        rows.append(
            {
                "image_id": items_df.iloc[i]["image_id"],
                "file_name": items_df.iloc[i]["file_name"],
                "file_path": items_df.iloc[i]["file_path"],
                "route": route,
                "final_leaf": final_leaf,
                "predicted_leaf": composed_leaf,
                "predicted_base_scene": base_scene,
                "predicted_orientation": parts[0] if len(parts) > 0 else "",
                "predicted_framing": parts[1] if len(parts) > 1 else "",
                "predicted_angle": parts[2] if len(parts) > 2 else "",
                "predicted_modifier": modifier,
                "confidence": round(conf, 4),
                "top2_base_scene": str(top2_labels[i]),
                "top2_confidence": round(float(top2_probs[i]), 4),
                "margin": round(margin, 4),
                "sitting_prob": round(sp, 4),
                "twisted_prob": round(tp, 4),
                "centroid_sim": round(centroid_sim, 4) if not math.isnan(centroid_sim) else None,
                "reason": reason_str,
            }
        )

    pred_df = pd.DataFrame(rows)

    # オプション: DINOv2 葉内重複除外（classified 対象）
    if bool(fcfg.get("enable_dedup", False)):
        dedup_thr = float(fcfg.get("max_leaf_pair_sim_dedup", 0.96))
        classified_mask = pred_df["route"] == "classified"
        for leaf, grp in pred_df[classified_mask].groupby("final_leaf"):
            if len(grp) <= 1:
                continue
            # 信頼度が高い順に保持し、既存保持画像と類似度 >= dedup_thr のものを duplicate へ
            grp_sorted = grp.sort_values("confidence", ascending=False)
            kept_indices: list[int] = []
            for row_idx in grp_sorted.index.tolist():
                emb = dino_embs[row_idx]
                is_dup = False
                for k_idx in kept_indices:
                    sim = float(np.dot(emb, dino_embs[k_idx]))
                    if sim >= dedup_thr:
                        pred_df.at[row_idx, "route"] = "duplicate"
                        pred_df.at[row_idx, "final_leaf"] = "_duplicate"
                        pred_df.at[row_idx, "reason"] = (
                            f"dup_of_{pred_df.at[k_idx, 'file_name']}(sim={sim:.3f})"
                        )
                        is_dup = True
                        break
                if not is_dup:
                    kept_indices.append(row_idx)

    summary_df = (
        pred_df.groupby(["route", "final_leaf"])
        .agg(
            count=("image_id", "count"),
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
        )
        .reset_index()
        .sort_values(["route", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary_df["mean_confidence"] = summary_df["mean_confidence"].round(4)
    summary_df["min_confidence"] = summary_df["min_confidence"].round(4)
    return pred_df, summary_df


def export_scene_predictions(
    pred_df: pd.DataFrame,
    output_dir: str,
    clear_export_dirs: bool = True,
    prefix_parent_dir: bool = True,
) -> dict:
    """
    推論結果をフォルダ構成ごと書き出し、CSV レポートを保存する。
    複数サブフォルダ（例: ComfyUI の日時フォルダ）を同時に処理してもファイル名が衝突しないよう、
    `prefix_parent_dir=True` の場合は `<親フォルダ名>__<元ファイル名>` で保存する。
    """
    out_root = Path(output_dir)
    classified_dir = out_root / "classified"
    uncertain_dir = out_root / "_uncertain"
    rejected_dir = out_root / "_rejected"
    duplicate_dir = out_root / "_duplicate"

    if clear_export_dirs:
        for d in (classified_dir, uncertain_dir, rejected_dir, duplicate_dir):
            if d.exists():
                shutil.rmtree(d)

    out_root.mkdir(parents=True, exist_ok=True)
    classified_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, int] = {}

    for row in pred_df.itertuples(index=False):
        src = Path(row.file_path)
        if prefix_parent_dir and src.parent.name:
            base_out_name = f"{src.parent.name}__{src.name}"
        else:
            base_out_name = src.name

        if row.route == "classified":
            dst_dir = classified_dir / row.final_leaf
        elif row.route == "uncertain":
            dst_dir = uncertain_dir
        elif row.route == "duplicate":
            dst_dir = duplicate_dir
        else:
            sub = row.final_leaf.replace("_rejected/", "", 1)
            dst_dir = rejected_dir / sub

        dst_dir.mkdir(parents=True, exist_ok=True)
        key = str(dst_dir / base_out_name)
        if key in used_names:
            used_names[key] += 1
            stem = Path(base_out_name).stem
            ext = Path(base_out_name).suffix
            dst_path = dst_dir / f"{stem}__{used_names[key]}{ext}"
        else:
            used_names[key] = 0
            dst_path = dst_dir / base_out_name

        shutil.copy2(src, dst_path)

    pred_csv = str(out_root / "scene_predictions.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    summary_df = (
        pred_df.groupby(["route", "final_leaf"])
        .agg(
            count=("image_id", "count"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["route", "count"], ascending=[True, False])
    )
    summary_csv = str(out_root / "scene_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    route_counts = pred_df["route"].value_counts().to_dict()
    print(f"✅ 書き出し完了: {output_dir} | 内訳: {route_counts}")
    return {
        "output_dir": str(out_root),
        "classified_dir": str(classified_dir),
        "uncertain_dir": str(uncertain_dir),
        "rejected_dir": str(rejected_dir),
        "duplicate_dir": str(duplicate_dir),
        "predictions_csv": pred_csv,
        "summary_csv": summary_csv,
    }


def show_scene_distribution_chart(summary_df: pd.DataFrame, top_n: int = 30) -> None:
    """分類先シーンの上位分布を横棒グラフで可視化する。"""
    sub = summary_df[summary_df["route"] == "classified"].head(top_n)
    if len(sub) == 0:
        print("classified の画像がありません。")
        return
    fig_h = max(4.0, len(sub) * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub["count"].values, color="#4C78A8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["final_leaf"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_title(f"Top {len(sub)} Classified Scenes")
    for i, (cnt, conf) in enumerate(zip(sub["count"].values, sub["mean_confidence"].values)):
        ax.text(cnt + 0.5, i, f"{cnt} (conf={conf:.2f})", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()


def show_scene_thumbnails(
    pred_df: pd.DataFrame,
    leaf: str | None = None,
    route: str | None = None,
    n: int = 12,
    cols: int = 4,
    sort_by: str = "confidence_asc",
) -> None:
    """指定シーンまたは指定ルート（例: route='uncertain'）のサムネイルを一覧表示する。"""
    sub = pred_df.copy()
    if route is not None:
        sub = sub[sub["route"] == route]
    if leaf is not None:
        sub = sub[sub["final_leaf"] == leaf]
    if len(sub) == 0:
        print(f"該当画像がありません (route={route}, leaf={leaf})")
        return

    if sort_by == "confidence_asc":
        sub = sub.sort_values("confidence", ascending=True)
    elif sort_by == "confidence_desc":
        sub = sub.sort_values("confidence", ascending=False)

    sub = sub.head(n)
    rows_n = math.ceil(len(sub) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.6, rows_n * 4.2))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, row in zip(axes_flat, sub.itertuples(index=False)):
        img = Image.open(row.file_path).convert("RGB")
        ax.imshow(img)
        title = (
            f"{row.file_name}\n"
            f"pred: {row.predicted_leaf}\n"
            f"conf={row.confidence:.2f} (top2={row.top2_confidence:.2f})"
        )
        if row.reason:
            title += f"\n[{row.reason[:36]}]"
        ax.set_title(title, fontsize=8)

    plt.tight_layout()
    plt.show()


# ============================================================================
# v2: 空間パッチ特徴 (CLS + 左右差分 + 上下差分) & 水平フリップ拡張 / TTA
# ============================================================================

_HFLIP_ORIENTATION_MAP: dict[str, str] = {
    "side_left": "side_right",
    "side_right": "side_left",
    "three_quarter_left": "three_quarter_right",
    "three_quarter_right": "three_quarter_left",
    "looking_back_left": "looking_back_right",
    "looking_back_right": "looking_back_left",
    "front": "front",
}


def flip_scene_label(label: str) -> str:
    """
    `orientation/framing/angle` 形式のラベルの左右 (`left` <-> `right`) を反転する。
    `front` 等はそのまま返す。
    """
    parts = str(label).split("/")
    if not parts:
        return str(label)
    parts[0] = _HFLIP_ORIENTATION_MAP.get(parts[0], parts[0])
    return "/".join(parts)


def _is_lr_mirror_error(true_label: str, pred_label: str) -> bool:
    """正解と予測が左右反転ペア（例: looking_back_left <-> looking_back_right）か判定する。"""
    t_ori = str(true_label).split("/")[0]
    p_ori = str(pred_label).split("/")[0]
    return t_ori != p_ori and _HFLIP_ORIENTATION_MAP.get(t_ori) == p_ori


def _read_image_and_md5_onepass(file_path: str) -> tuple[np.ndarray, str]:
    """
    Drive FUSE 越しのネットワーク往復を半減させるため、
    1 回の read_bytes() から md5 計算と cv2.imdecode() を同時に行う。
    """
    raw_bytes = Path(file_path).read_bytes()
    md5 = hashlib.md5(raw_bytes).hexdigest()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise RuntimeError(f"画像のデコードに失敗しました: {file_path}")
    return img_bgr, md5


def extract_or_load_scene_feature_pack(
    items_df: pd.DataFrame,
    cache_dir: str,
    embedder,
    tagger=None,
    include_hflip: bool = True,
    force_recompute: bool = False,
    cache_prefix: str = "scene_features",
    progress_every: int = 100,
) -> dict:
    """
    1 パス読み込みで DINOv2 空間特徴 (`cls`, `lr_diff`, `tb_diff`)、
    水平フリップ DINOv2 空間特徴 (`*_flip`)、および WD14-Tagger 全 general タグスコアを抽出・キャッシュする。

    - v2 キャッシュが既に全件ヒットすれば即座にロードする（`image_id:file_size` による高速照合対応）。
    - 同ディレクトリに v1 キャッシュ (`{cache_prefix}.npz`) が存在する場合、
      WD14 タグスコアは v1 キャッシュから自動流用し、WD14 の再推論をスキップする。
    """
    os.makedirs(cache_dir, exist_ok=True)
    v2_npz = os.path.join(cache_dir, f"{cache_prefix}_v2.npz")
    v2_meta = os.path.join(cache_dir, f"{cache_prefix}_v2_meta.json")
    v1_npz = os.path.join(cache_dir, f"{cache_prefix}.npz")
    v1_meta = os.path.join(cache_dir, f"{cache_prefix}_meta.json")

    file_paths = items_df["file_path"].astype(str).tolist()
    image_ids = (
        items_df["image_id"].astype(str).tolist()
        if "image_id" in items_df.columns
        else [Path(p).name for p in file_paths]
    )
    known_md5s = (
        items_df["md5"].astype(str).tolist() if "md5" in items_df.columns else None
    )
    want_wd14 = tagger is not None

    # 1. 高速キャッシュヒット判定（v2 キャッシュ）
    if not force_recompute and os.path.exists(v2_npz) and os.path.exists(v2_meta):
        with open(v2_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        has_wd14 = bool(meta.get("has_wd14", False))
        has_hflip = bool(meta.get("has_hflip", False))

        if (not want_wd14 or has_wd14) and (not include_hflip or has_hflip):
            order = None
            if known_md5s is not None and "md5_list" in meta:
                md5_map = {m: i for i, m in enumerate(meta["md5_list"])}
                if set(known_md5s).issubset(md5_map.keys()):
                    order = [md5_map[m] for m in known_md5s]
            elif "fast_keys" in meta:
                fast_keys = [
                    f"{iid}:{os.path.getsize(p)}" for iid, p in zip(image_ids, file_paths)
                ]
                fk_map = {k: i for i, k in enumerate(meta["fast_keys"])}
                if set(fast_keys).issubset(fk_map.keys()):
                    order = [fk_map[k] for k in fast_keys]

            if order is not None:
                data = np.load(v2_npz, allow_pickle=False)
                pack = {
                    "dino_cls": data["dino_cls"][order].astype(np.float32),
                    "dino_lr_diff": data["dino_lr_diff"][order].astype(np.float32),
                    "dino_tb_diff": data["dino_tb_diff"][order].astype(np.float32),
                    "dino_cls_flip": (
                        data["dino_cls_flip"][order].astype(np.float32)
                        if include_hflip and "dino_cls_flip" in data
                        else None
                    ),
                    "dino_lr_diff_flip": (
                        data["dino_lr_diff_flip"][order].astype(np.float32)
                        if include_hflip and "dino_lr_diff_flip" in data
                        else None
                    ),
                    "dino_tb_diff_flip": (
                        data["dino_tb_diff_flip"][order].astype(np.float32)
                        if include_hflip and "dino_tb_diff_flip" in data
                        else None
                    ),
                    "wd14_probs": (
                        data["wd14_probs"][order].astype(np.float32)
                        if want_wd14 and "wd14_probs" in data
                        else None
                    ),
                    "wd14_tag_names": list(meta.get("wd14_tag_names", [])) if want_wd14 else [],
                }
                print(
                    f"✅ v2 キャッシュから特徴量を読み込みました ({len(order)} 枚 | "
                    f"spatial=True, hflip={pack['dino_cls_flip'] is not None}, "
                    f"WD14={None if pack['wd14_probs'] is None else pack['wd14_probs'].shape})"
                )
                return pack

    # 2. v1 キャッシュから WD14 のみ流用可能かチェック（WD14 再推論を大幅節約）
    v1_wd14_by_md5: dict[str, np.ndarray] = {}
    wd14_tag_names: list[str] = (
        [tagger.tag_names[i] for i in tagger.general_indices] if want_wd14 else []
    )
    if want_wd14 and not force_recompute and os.path.exists(v1_npz) and os.path.exists(v1_meta):
        with open(v1_meta, "r", encoding="utf-8") as f:
            m1 = json.load(f)
        if m1.get("has_wd14") and list(m1.get("wd14_tag_names", [])) == wd14_tag_names:
            d1 = np.load(v1_npz, allow_pickle=False)
            if "wd14_probs" in d1:
                for i, m in enumerate(m1.get("md5_list", [])):
                    v1_wd14_by_md5[m] = d1["wd14_probs"][i]
                print(
                    f"ℹ️ 既存 v1 キャッシュから WD14 スコアを自動流用します ({len(v1_wd14_by_md5)} 件利用可能)"
                )

    # 3. 1 パス読み込み + GPU 推論
    cls_rows, lr_rows, tb_rows = [], [], []
    cls_f_rows, lr_f_rows, tb_f_rows = [], [], []
    wd14_rows = []
    md5_out = []
    fast_keys_out = []

    n_total = len(file_paths)
    n_wd14_reused = 0
    t_start = time.perf_counter()

    for idx, (iid, fpath) in enumerate(zip(image_ids, file_paths), start=1):
        img_bgr, md5 = _read_image_and_md5_onepass(fpath)
        md5_out.append(md5)
        fast_keys_out.append(f"{iid}:{os.path.getsize(fpath)}")

        sp = embedder.embed_spatial(img_bgr)
        cls_rows.append(sp["cls"])
        lr_rows.append(sp["lr_diff"])
        tb_rows.append(sp["tb_diff"])

        if include_hflip:
            img_flip = cv2.flip(img_bgr, 1)
            sp_f = embedder.embed_spatial(img_flip)
            cls_f_rows.append(sp_f["cls"])
            lr_f_rows.append(sp_f["lr_diff"])
            tb_f_rows.append(sp_f["tb_diff"])

        if want_wd14:
            if md5 in v1_wd14_by_md5:
                wd14_rows.append(v1_wd14_by_md5[md5].astype(np.float16))
                n_wd14_reused += 1
            else:
                pred = tagger.predict(img_bgr)["general"]
                vec = np.array([pred[t] for t in wd14_tag_names], dtype=np.float16)
                wd14_rows.append(vec)

        if progress_every and (idx % progress_every == 0 or idx == n_total):
            elapsed = time.perf_counter() - t_start
            rate = idx / max(elapsed, 1e-6)
            eta = (n_total - idx) / max(rate, 1e-6)
            print(
                f"  v2 特徴抽出進捗: {idx}/{n_total} 枚 "
                f"({elapsed:.1f}s 経過, 残り約 {eta:.0f}s | WD14再利用: {n_wd14_reused} 枚)"
            )

    dino_cls = np.vstack(cls_rows).astype(np.float32)
    dino_lr = np.vstack(lr_rows).astype(np.float32)
    dino_tb = np.vstack(tb_rows).astype(np.float32)

    save_dict = {
        "dino_cls": dino_cls,
        "dino_lr_diff": dino_lr,
        "dino_tb_diff": dino_tb,
    }
    if include_hflip:
        dino_cls_f = np.vstack(cls_f_rows).astype(np.float32)
        dino_lr_f = np.vstack(lr_f_rows).astype(np.float32)
        dino_tb_f = np.vstack(tb_f_rows).astype(np.float32)
        save_dict["dino_cls_flip"] = dino_cls_f
        save_dict["dino_lr_diff_flip"] = dino_lr_f
        save_dict["dino_tb_diff_flip"] = dino_tb_f
    else:
        dino_cls_f = dino_lr_f = dino_tb_f = None

    if want_wd14:
        wd14_f16 = np.vstack(wd14_rows).astype(np.float16)
        save_dict["wd14_probs"] = wd14_f16
        wd14_probs = wd14_f16.astype(np.float32)
    else:
        wd14_probs = None

    np.savez_compressed(v2_npz, **save_dict)
    meta_out = {
        "version": "feature_pack_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_total,
        "md5_list": md5_out,
        "fast_keys": fast_keys_out,
        "has_hflip": include_hflip,
        "has_wd14": want_wd14,
        "wd14_tag_names": wd14_tag_names,
        "dinov2_model_id": getattr(embedder, "model_id", None),
        "wd14_repo_id": getattr(tagger, "repo_id", None) if want_wd14 else None,
    }
    with open(v2_meta, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"💾 v2 特徴量キャッシュ保存完了: {v2_npz}")
    return {
        "dino_cls": dino_cls,
        "dino_lr_diff": dino_lr,
        "dino_tb_diff": dino_tb,
        "dino_cls_flip": dino_cls_f,
        "dino_lr_diff_flip": dino_lr_f,
        "dino_tb_diff_flip": dino_tb_f,
        "wd14_probs": wd14_probs,
        "wd14_tag_names": wd14_tag_names,
    }


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    arr = mat.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


def combine_scene_features_v2(
    feature_pack: dict,
    indices: np.ndarray | list[int] | None = None,
    flipped: bool = False,
    selected_wd14_indices: list[int] | None = None,
    feature_mode: str = "dino_spatial_wd14",
    spatial_weight: float = 0.7,
    wd14_weight: float = 1.0,
) -> np.ndarray:
    """
    v2 特徴パックから指定モードの特徴行列を組み立てる。
    - `dino_only`: [cls]
    - `dino_wd14`: [cls, wd14]
    - `dino_spatial`: [cls, lr_diff * spatial_weight, tb_diff * spatial_weight]
    - `dino_spatial_wd14`: [cls, lr_diff * spatial_weight, tb_diff * spatial_weight, wd14 * wd14_weight]
    """
    cls_key = "dino_cls_flip" if flipped else "dino_cls"
    lr_key = "dino_lr_diff_flip" if flipped else "dino_lr_diff"
    tb_key = "dino_tb_diff_flip" if flipped else "dino_tb_diff"

    cls_mat = feature_pack[cls_key]
    if indices is not None:
        cls_mat = cls_mat[indices]
    blocks = [_l2_normalize_rows(cls_mat)]

    if "spatial" in feature_mode:
        lr_mat = feature_pack[lr_key]
        tb_mat = feature_pack[tb_key]
        if indices is not None:
            lr_mat = lr_mat[indices]
            tb_mat = tb_mat[indices]
        blocks.append(_l2_normalize_rows(lr_mat) * float(spatial_weight))
        blocks.append(_l2_normalize_rows(tb_mat) * float(spatial_weight))

    if "wd14" in feature_mode and feature_pack.get("wd14_probs") is not None:
        w_mat = feature_pack["wd14_probs"]
        if indices is not None:
            w_mat = w_mat[indices]
        if selected_wd14_indices is not None:
            w_mat = w_mat[:, selected_wd14_indices]
        blocks.append(_l2_normalize_rows(w_mat) * float(wd14_weight))

    return np.hstack(blocks).astype(np.float32)


def evaluate_scene_classifiers_v2(
    train_df: pd.DataFrame,
    feature_pack: dict,
    feature_modes: list[str] | None = None,
    model_names: list[str] | None = None,
    hflip_aug_modes: list[bool] | None = None,
    use_hflip_tta: bool = True,
    n_splits: int = 5,
    spatial_weight: float = 0.7,
    wd14_min_max_score: float = 0.15,
    wd14_max_tags: int = 512,
    wd14_weight: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    v2 特徴量（空間パッチ特徴 + 水平フリップ 2 倍学習 + 推論時 HFlip TTA）を
    リークゼロの Stratified K-Fold CV で比較評価する。
    `orientation_acc` と `lr_confusions`（左右反転ミス件数）も算出する。
    """
    wd14_probs = feature_pack.get("wd14_probs")
    wd14_tag_names = feature_pack.get("wd14_tag_names", [])
    has_hflip = feature_pack.get("dino_cls_flip") is not None

    if feature_modes is None:
        feature_modes = (
            ["dino_wd14", "dino_spatial_wd14"]
            if wd14_probs is not None
            else ["dino_only", "dino_spatial"]
        )
    if model_names is None:
        model_names = ["logreg_strong", "logreg_c30", "logreg_balanced", "linear_svc_cal"]
    if hflip_aug_modes is None:
        hflip_aug_modes = [False, True] if has_hflip else [False]

    y_base = train_df["base_scene"].astype(str).to_numpy()
    y_base_flip = np.array([flip_scene_label(c) for c in y_base], dtype=object)
    y_sit = train_df["is_sitting"].astype(int).to_numpy()
    y_twi = train_df["is_twisted"].astype(int).to_numpy()

    counts = pd.Series(y_base).value_counts()
    eval_mask = np.array([counts[c] >= n_splits for c in y_base])
    eval_indices = np.where(eval_mask)[0]
    rare_indices = np.where(~eval_mask)[0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(eval_indices, y_base[eval_indices]))

    skf_sit = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_sit = list(skf_sit.split(np.arange(len(y_sit)), y_sit))
    skf_twi = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_twi = list(skf_twi.split(np.arange(len(y_twi)), y_twi))

    rows = []
    best_score = -1.0
    best_oof: dict = {}

    for fmode in feature_modes:
        if "wd14" in fmode and wd14_probs is None:
            continue

        for use_hflip in hflip_aug_modes:
            if use_hflip and not has_hflip:
                continue

            # fold ごとの特徴行列を事前構築
            base_mats = []
            for a, b in folds:
                tr_idx = np.concatenate([eval_indices[a], rare_indices])
                va_idx = eval_indices[b]
                sel_idx = (
                    select_wd14_active_tags(
                        wd14_probs[tr_idx],
                        wd14_tag_names,
                        min_max_score=wd14_min_max_score,
                        max_tags=wd14_max_tags,
                    )[0]
                    if ("wd14" in fmode and wd14_probs is not None)
                    else None
                )
                X_tr = combine_scene_features_v2(
                    feature_pack, tr_idx, False, sel_idx, fmode, spatial_weight, wd14_weight
                )
                y_tr = y_base[tr_idx]
                if use_hflip:
                    X_tr_f = combine_scene_features_v2(
                        feature_pack, tr_idx, True, sel_idx, fmode, spatial_weight, wd14_weight
                    )
                    X_tr = np.vstack([X_tr, X_tr_f])
                    y_tr = np.concatenate([y_tr, y_base_flip[tr_idx]])

                X_va = combine_scene_features_v2(
                    feature_pack, va_idx, False, sel_idx, fmode, spatial_weight, wd14_weight
                )
                X_va_f = (
                    combine_scene_features_v2(
                        feature_pack, va_idx, True, sel_idx, fmode, spatial_weight, wd14_weight
                    )
                    if (use_hflip_tta and has_hflip)
                    else None
                )
                base_mats.append((b, X_tr, y_tr, X_va, X_va_f))

            sit_mats = []
            for tr_idx, va_idx in folds_sit:
                sel_idx = (
                    select_wd14_active_tags(
                        wd14_probs[tr_idx], wd14_tag_names, wd14_min_max_score, wd14_max_tags
                    )[0]
                    if ("wd14" in fmode and wd14_probs is not None)
                    else None
                )
                X_tr = combine_scene_features_v2(
                    feature_pack, tr_idx, False, sel_idx, fmode, spatial_weight, wd14_weight
                )
                y_tr = y_sit[tr_idx]
                if use_hflip:
                    X_tr = np.vstack([
                        X_tr,
                        combine_scene_features_v2(
                            feature_pack, tr_idx, True, sel_idx, fmode, spatial_weight, wd14_weight
                        ),
                    ])
                    y_tr = np.concatenate([y_tr, y_sit[tr_idx]])
                X_va = combine_scene_features_v2(
                    feature_pack, va_idx, False, sel_idx, fmode, spatial_weight, wd14_weight
                )
                sit_mats.append((va_idx, X_tr, y_tr, X_va))

            twi_mats = []
            for tr_idx, va_idx in folds_twi:
                sel_idx = (
                    select_wd14_active_tags(
                        wd14_probs[tr_idx], wd14_tag_names, wd14_min_max_score, wd14_max_tags
                    )[0]
                    if ("wd14" in fmode and wd14_probs is not None)
                    else None
                )
                X_tr = combine_scene_features_v2(
                    feature_pack, tr_idx, False, sel_idx, fmode, spatial_weight, wd14_weight
                )
                y_tr = y_twi[tr_idx]
                if use_hflip:
                    X_tr = np.vstack([
                        X_tr,
                        combine_scene_features_v2(
                            feature_pack, tr_idx, True, sel_idx, fmode, spatial_weight, wd14_weight
                        ),
                    ])
                    y_tr = np.concatenate([y_tr, y_twi[tr_idx]])
                X_va = combine_scene_features_v2(
                    feature_pack, va_idx, False, sel_idx, fmode, spatial_weight, wd14_weight
                )
                twi_mats.append((va_idx, X_tr, y_tr, X_va))

            binary_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

            for mname in model_names:
                t0 = time.perf_counter()
                oof_pred = np.empty(len(eval_indices), dtype=object)
                oof_top1_prob = np.zeros(len(eval_indices), dtype=np.float32)
                oof_top2_label = np.empty(len(eval_indices), dtype=object)
                oof_top2_prob = np.zeros(len(eval_indices), dtype=np.float32)
                eval_classes = sorted(np.unique(y_base[eval_indices]).tolist())
                class_to_col = {c: i for i, c in enumerate(eval_classes)}
                oof_prob_mat = np.zeros((len(eval_indices), len(eval_classes)), dtype=np.float32)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for va_rel, X_tr, y_tr, X_va, X_va_f in base_mats:
                        clf = _build_estimator(mname, random_state=random_state)
                        clf.fit(X_tr, y_tr)
                        probs = clf.predict_proba(X_va)
                        clf_classes = clf.classes_

                        if X_va_f is not None:
                            probs_f = clf.predict_proba(X_va_f)
                            c_to_col = {str(c): idx for idx, c in enumerate(clf_classes)}
                            aligned = np.zeros_like(probs_f)
                            for c_idx, c_name in enumerate(clf_classes):
                                dst = c_to_col.get(flip_scene_label(str(c_name)))
                                if dst is not None:
                                    aligned[:, dst] += probs_f[:, c_idx]
                                else:
                                    aligned[:, c_idx] += probs_f[:, c_idx]
                            probs = 0.5 * (probs + aligned)

                        order = np.argsort(-probs, axis=1)
                        top1_idx = order[:, 0]
                        top2_idx = order[:, 1] if probs.shape[1] > 1 else order[:, 0]

                        oof_pred[va_rel] = clf_classes[top1_idx]
                        oof_top1_prob[va_rel] = probs[np.arange(len(va_rel)), top1_idx]
                        oof_top2_label[va_rel] = clf_classes[top2_idx]
                        oof_top2_prob[va_rel] = probs[np.arange(len(va_rel)), top2_idx]

                        for c_pos, c_name in enumerate(clf_classes):
                            if c_name in class_to_col:
                                oof_prob_mat[va_rel, class_to_col[c_name]] = probs[:, c_pos]

                    bin_key = mname if mname in ("mlp", "linear_svc_cal") else "logreg"
                    if bin_key in binary_cache:
                        oof_sit_prob, oof_twi_prob = binary_cache[bin_key]
                    else:
                        oof_sit_prob = np.zeros(len(y_sit), dtype=np.float32)
                        for va_idx, X_tr, y_tr, X_va in sit_mats:
                            clf_s = _build_binary_estimator(mname, random_state=random_state)
                            clf_s.fit(X_tr, y_tr)
                            oof_sit_prob[va_idx] = clf_s.predict_proba(X_va)[:, 1]

                        oof_twi_prob = np.zeros(len(y_twi), dtype=np.float32)
                        for va_idx, X_tr, y_tr, X_va in twi_mats:
                            clf_t = _build_binary_estimator(mname, random_state=random_state)
                            clf_t.fit(X_tr, y_tr)
                            oof_twi_prob[va_idx] = clf_t.predict_proba(X_va)[:, 1]
                        binary_cache[bin_key] = (oof_sit_prob, oof_twi_prob)

                elapsed = time.perf_counter() - t0
                y_true_eval = y_base[eval_indices]
                acc = float(accuracy_score(y_true_eval, oof_pred))
                macro_f1 = float(
                    f1_score(y_true_eval, oof_pred, average="macro", zero_division=0)
                )
                weighted_f1 = float(
                    f1_score(y_true_eval, oof_pred, average="weighted", zero_division=0)
                )
                ori_acc = float(
                    np.mean(
                        [
                            t.split("/")[0] == p.split("/")[0]
                            for t, p in zip(y_true_eval, oof_pred)
                        ]
                    )
                )
                lr_conf = int(
                    sum(
                        1
                        for t, p in zip(y_true_eval, oof_pred)
                        if _is_lr_mirror_error(t, p)
                    )
                )

                row_sums = oof_prob_mat.sum(axis=1, keepdims=True)
                norm_prob_mat = oof_prob_mat / np.clip(row_sums, 1e-8, None)
                y_true_col = np.array([class_to_col[c] for c in y_true_eval])
                top3 = float(
                    top_k_accuracy_score(
                        y_true_col,
                        norm_prob_mat,
                        k=min(3, len(eval_classes)),
                        labels=np.arange(len(eval_classes)),
                    )
                )
                sit_f1 = float(f1_score(y_sit, (oof_sit_prob >= 0.5).astype(int), zero_division=0))
                sit_ap = float(average_precision_score(y_sit, oof_sit_prob)) if y_sit.sum() > 0 else 0.0
                twi_f1 = float(f1_score(y_twi, (oof_twi_prob >= 0.5).astype(int), zero_division=0))
                twi_ap = float(average_precision_score(y_twi, oof_twi_prob)) if y_twi.sum() > 0 else 0.0

                row = {
                    "feature_mode": fmode,
                    "hflip_aug": bool(use_hflip),
                    "model_name": mname,
                    "accuracy": round(acc, 4),
                    "macro_f1": round(macro_f1, 4),
                    "weighted_f1": round(weighted_f1, 4),
                    "top3_accuracy": round(top3, 4),
                    "orientation_acc": round(ori_acc, 4),
                    "lr_confusions": lr_conf,
                    "sitting_f1": round(sit_f1, 4),
                    "sitting_ap": round(sit_ap, 4),
                    "twisted_f1": round(twi_f1, 4),
                    "twisted_ap": round(twi_ap, 4),
                    "elapsed_sec": round(elapsed, 1),
                }
                rows.append(row)
                print(
                    f"[{fmode:17s} | flip={str(use_hflip):5s} | {mname:15s}] ({elapsed:5.1f}s) "
                    f"acc={acc:.4f}, macro_f1={macro_f1:.4f}, ori={ori_acc:.4f}, LR_err={lr_conf:3d}, "
                    f"top3={top3:.4f} | sit_f1={sit_f1:.4f}, twi_f1={twi_f1:.4f}"
                )

                composite = 0.5 * macro_f1 + 0.5 * acc
                if composite > best_score:
                    best_score = composite
                    oof_df = train_df.iloc[eval_indices].copy().reset_index(drop=True)
                    oof_df["oof_pred"] = oof_pred
                    oof_df["oof_prob"] = oof_top1_prob
                    oof_df["oof_top2_label"] = oof_top2_label
                    oof_df["oof_top2_prob"] = oof_top2_prob
                    oof_df["oof_margin"] = oof_top1_prob - oof_top2_prob
                    oof_df["oof_correct"] = oof_df["base_scene"] == oof_df["oof_pred"]
                    oof_df["oof_sitting_prob"] = oof_sit_prob[eval_indices]
                    oof_df["oof_twisted_prob"] = oof_twi_prob[eval_indices]

                    wrong = oof_df[~oof_df["oof_correct"]]
                    conf_pairs = (
                        wrong.groupby(["base_scene", "oof_pred"])
                        .size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                        .reset_index(drop=True)
                    )
                    best_oof = {
                        "config": {
                            "feature_mode": fmode,
                            "use_hflip_aug": bool(use_hflip),
                            "model_name": mname,
                            "spatial_weight": float(spatial_weight),
                            "wd14_min_max_score": float(wd14_min_max_score),
                            "wd14_max_tags": int(wd14_max_tags),
                            "wd14_weight": float(wd14_weight),
                        },
                        "metrics": row,
                        "oof_df": oof_df,
                        "confusion_pairs": conf_pairs,
                    }

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    return comparison_df, best_oof


def train_scene_classifier_bundle_v2(
    train_df: pd.DataFrame,
    feature_pack: dict,
    feature_mode: str = "dino_spatial_wd14",
    model_name: str = "logreg_c30",
    use_hflip_aug: bool = True,
    spatial_weight: float = 0.7,
    wd14_min_max_score: float = 0.15,
    wd14_max_tags: int = 512,
    wd14_weight: float = 1.0,
    dinov2_model_id: str | None = None,
    wd14_repo_id: str | None = None,
    cv_metrics: dict | None = None,
    random_state: int = 42,
) -> dict:
    """
    v2 特徴量（空間パッチ特徴 + 水平フリップ 2 倍学習）で本番モデルバンドルを学習する。
    """
    wd14_probs = feature_pack.get("wd14_probs")
    wd14_tag_names = feature_pack.get("wd14_tag_names", [])

    if "wd14" in feature_mode and wd14_probs is not None:
        selected_wd14_indices, selected_wd14_names = select_wd14_active_tags(
            wd14_probs,
            wd14_tag_names,
            min_max_score=wd14_min_max_score,
            max_tags=wd14_max_tags,
        )
    else:
        selected_wd14_indices = []
        selected_wd14_names = []

    X = combine_scene_features_v2(
        feature_pack=feature_pack,
        flipped=False,
        selected_wd14_indices=selected_wd14_indices if selected_wd14_indices else None,
        feature_mode=feature_mode,
        spatial_weight=spatial_weight,
        wd14_weight=wd14_weight,
    )
    y_base = train_df["base_scene"].astype(str).to_numpy()
    y_sit = train_df["is_sitting"].astype(int).to_numpy()
    y_twi = train_df["is_twisted"].astype(int).to_numpy()

    dino_cls_all = feature_pack["dino_cls"]
    y_base_all = y_base

    if use_hflip_aug and feature_pack.get("dino_cls_flip") is not None:
        X_flip = combine_scene_features_v2(
            feature_pack=feature_pack,
            flipped=True,
            selected_wd14_indices=selected_wd14_indices if selected_wd14_indices else None,
            feature_mode=feature_mode,
            spatial_weight=spatial_weight,
            wd14_weight=wd14_weight,
        )
        y_base_flip = np.array([flip_scene_label(c) for c in y_base], dtype=object)
        X = np.vstack([X, X_flip])
        y_base_all = np.concatenate([y_base, y_base_flip])
        y_sit_all = np.concatenate([y_sit, y_sit])
        y_twi_all = np.concatenate([y_twi, y_twi])
        dino_cls_all = np.vstack([feature_pack["dino_cls"], feature_pack["dino_cls_flip"]])
    else:
        y_sit_all = y_sit
        y_twi_all = y_twi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_clf = _build_estimator(model_name, random_state=random_state)
        base_clf.fit(X, y_base_all)

        sitting_clf = None
        if len(np.unique(y_sit_all)) >= 2:
            sitting_clf = _build_binary_estimator(model_name, random_state=random_state)
            sitting_clf.fit(X, y_sit_all)

        twisted_clf = None
        if len(np.unique(y_twi_all)) >= 2:
            twisted_clf = _build_binary_estimator(model_name, random_state=random_state)
            twisted_clf.fit(X, y_twi_all)

    class_centroids: dict[str, np.ndarray] = {}
    for cls in sorted(np.unique(y_base_all).tolist()):
        embs = dino_cls_all[y_base_all == cls]
        mean_vec = embs.mean(axis=0).astype(np.float32)
        mean_vec = mean_vec / max(float(np.linalg.norm(mean_vec)), 1e-8)
        class_centroids[cls] = mean_vec

    class_counts = {str(k): int(v) for k, v in pd.Series(y_base_all).value_counts().items()}

    bundle = {
        "version": "scene_classifier_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_mode": feature_mode,
        "model_name": model_name,
        "use_hflip_aug": bool(use_hflip_aug),
        "spatial_weight": float(spatial_weight),
        "wd14_weight": float(wd14_weight),
        "wd14_min_max_score": float(wd14_min_max_score),
        "wd14_max_tags": int(wd14_max_tags),
        "selected_wd14_tag_names": selected_wd14_names,
        "dinov2_model_id": dinov2_model_id,
        "wd14_repo_id": wd14_repo_id,
        "base_classes": [str(c) for c in base_clf.classes_.tolist()],
        "class_counts": class_counts,
        "n_train_samples_orig": int(len(train_df)),
        "n_train_samples": int(len(y_base_all)),
        "n_sitting_pos": int(y_sit_all.sum()),
        "n_twisted_pos": int(y_twi_all.sum()),
        "cv_metrics": cv_metrics or {},
        "base_clf": base_clf,
        "sitting_clf": sitting_clf,
        "twisted_clf": twisted_clf,
        "class_centroids": class_centroids,
    }
    print(
        f"✅ v2 本番モデル学習完了: {len(bundle['base_classes'])} ベースクラス "
        f"({len(y_base_all)} サンプル [元 {len(train_df)} 枚, hflip={use_hflip_aug}] | "
        f"mode={feature_mode}, model={model_name}, wd14_tags={len(selected_wd14_names)})"
    )
    return bundle
