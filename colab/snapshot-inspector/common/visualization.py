"""Visualization and Google Drive export helpers for snapshot-inspector."""

from __future__ import annotations

import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _style_image_axis(ax, title: str, border_color: str = "gray"):
    """画像表示用の軸から目盛りを消し、枠線を表示する。"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)


def show_profile_summary(profile: dict, top_n: int = 20, figsize=(14, 6)):
    """
    構築したキャラクター固有分布（不変タグの平均±標準偏差 & プロポーション分布）を可視化する。
    """
    core_tags = profile.get("identity_core_tags", {})
    props = profile.get("proportions", {})

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    ax_tags = axes[0]
    items = list(core_tags.items())[:top_n]
    if items:
        names = [k for k, _ in items][::-1]
        means = [v["mean"] for _, v in items][::-1]
        stds = [v["std"] for _, v in items][::-1]
        y_pos = np.arange(len(names))
        ax_tags.barh(
            y_pos,
            means,
            xerr=stds,
            color="#4C78A8",
            alpha=0.85,
            ecolor="#333333",
            capsize=3,
        )
        ax_tags.set_yticks(y_pos)
        ax_tags.set_yticklabels(names, fontsize=9)
        ax_tags.set_xlim(0.0, 1.08)
        ax_tags.set_xlabel("WD14 Confidence (Mean +/- Std)")
        ax_tags.set_title(
            f"Identity Core Tags (n={profile['reference_count']}, total={len(core_tags)})"
        )
        ax_tags.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax_tags.text(0.5, 0.5, "No Identity Core Tags Found", ha="center", va="center")
        ax_tags.set_axis_off()

    ax_prop = axes[1]
    if props:
        p_names = list(props.keys())[::-1]
        p_means = [props[k]["mean"] for k in p_names]
        p_stds = [props[k]["std"] for k in p_names]
        y_pos = np.arange(len(p_names))
        ax_prop.barh(
            y_pos,
            p_means,
            xerr=p_stds,
            color="#72B7B2",
            alpha=0.85,
            ecolor="#333333",
            capsize=3,
        )
        ax_prop.set_yticks(y_pos)
        ax_prop.set_yticklabels(p_names, fontsize=9)
        ax_prop.set_xlabel("Scale-Invariant Ratio (Mean +/- Std)")
        ax_prop.set_title("Pose Proportions Distribution")
        ax_prop.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax_prop.text(
            0.5, 0.5, "No Pose Proportions Available", ha="center", va="center"
        )
        ax_prop.set_axis_off()

    plt.tight_layout()
    plt.show()


def show_target_report(
    record: dict, consistency: dict, debug_skel: np.ndarray, figsize=(15, 5)
):
    """
    対象画像1枚について、骨格プレビュー・表情/視線スコア・キャラ固有タグの一致状況を並べて表示する。
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(cv2.cvtColor(debug_skel, cv2.COLOR_BGR2RGB))
    ov = consistency.get("overall_consistency")
    ov_str = f"{ov:.3f}" if ov is not None else "N/A"
    _style_image_axis(axes[0], f"{record['file_name']} (Consistency: {ov_str})")

    ax_var = axes[1]
    slot_scores = []
    for slot_name in ["expression", "gaze", "framing", "pose_action"]:
        for tag, val in record["tags"]["slots"].get(slot_name, {}).items():
            if val >= 0.15:
                slot_scores.append((f"[{slot_name[:4]}] {tag}", val))
    slot_scores = sorted(slot_scores, key=lambda x: x[1], reverse=True)[:12]
    if slot_scores:
        labels = [k for k, _ in slot_scores][::-1]
        vals = [v for _, v in slot_scores][::-1]
        ax_var.barh(np.arange(len(labels)), vals, color="#F58518", alpha=0.85)
        ax_var.set_yticks(np.arange(len(labels)))
        ax_var.set_yticklabels(labels, fontsize=8.5)
        ax_var.set_xlim(0.0, 1.05)
        ax_var.set_xlabel("Confidence")
        ax_var.set_title("Active Variable Slots (>= 0.15)")
        ax_var.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax_var.text(0.5, 0.5, "No Active Slot Tags", ha="center", va="center")
        ax_var.set_axis_off()

    ax_id = axes[2]
    comp = consistency.get("core_tag_comparison", {})
    if comp:
        ranked = sorted(comp.items(), key=lambda kv: kv[1]["z_score"])[:12]
        labels = [k for k, _ in ranked][::-1]
        actuals = [v["actual"] for _, v in ranked][::-1]
        expected = [v["expected_mean"] for _, v in ranked][::-1]
        colors = ["#E45756" if v["alert"] else "#54A24B" for _, v in ranked][::-1]
        y_pos = np.arange(len(labels))
        ax_id.barh(y_pos, actuals, color=colors, alpha=0.85, label="Actual")
        ax_id.scatter(expected, y_pos, color="#222222", zorder=5, s=30, label="Ref Mean")
        ax_id.set_yticks(y_pos)
        ax_id.set_yticklabels(labels, fontsize=8.5)
        ax_id.set_xlim(0.0, 1.08)
        ax_id.set_xlabel("Confidence (Red = Weak/Missing Alert)")
        ax_id.set_title("Identity Core Check (Lowest z-score)")
        ax_id.legend(loc="lower right", fontsize=8)
        ax_id.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax_id.text(0.5, 0.5, "No Core Tag Comparison", ha="center", va="center")
        ax_id.set_axis_off()

    plt.tight_layout()
    plt.show()

    weak = consistency.get("missing_or_weak_traits", [])
    unexp = consistency.get("unexpected_traits", [])
    p_alerts = consistency.get("proportion_alerts", [])
    if weak:
        print(
            "  ⚠️ 固有特徴の低下・欠落:",
            ", ".join(
                f"{w['tag']} (実測 {w['actual']:.2f} vs 基準 {w['expected_mean']:.2f}±{w['expected_std']:.2f})"
                for w in weak
            ),
        )
    if unexp:
        print(
            "  ⚠️ 学習分布にない外乱タグ:",
            ", ".join(
                f"{u['tag']} ({u['actual']:.2f}, 基準平均 {u['reference_mean']:.2f})"
                for u in unexp[:5]
            ),
        )
    if p_alerts:
        print(
            "  ⚠️ 骨格プロポーション逸脱:",
            ", ".join(
                f"{p['metric']} (実測 {p['actual']:.3f} vs 基準 {p['expected_mean']:.3f}, z={p['z_score']:+.2f})"
                for p in p_alerts
            ),
        )
    if not weak and not unexp and not p_alerts:
        print("  ✅ キャラクター固有分布に対する大きな逸脱は検出されませんでした。")


def show_batch_variable_heatmap(
    records: list[dict],
    slot_name: str = "expression",
    min_max_score: float = 0.20,
    figsize=(12, 5),
):
    """
    複数画像（立ち絵差分・アニメ連番）における指定スロットの信頼度スコア推移をヒートマップで表示する。
    """
    if len(records) < 2:
        print("ℹ️ ヒートマップ比較には対象画像が2枚以上必要です。")
        return

    slot_tags = list(records[0]["tags"]["slots"].get(slot_name, {}).keys())
    matrix = np.array(
        [[r["tags"]["slots"][slot_name][t] for t in slot_tags] for r in records],
        dtype=np.float32,
    )

    keep = np.max(matrix, axis=0) >= min_max_score
    if not np.any(keep):
        print(
            f"ℹ️ スロット '{slot_name}' にしきい値 {min_max_score} 以上のタグがありません。"
        )
        return

    filtered_tags = [t for t, k in zip(slot_tags, keep) if k]
    filtered_mat = matrix[:, keep]
    img_labels = [r["image_id"] for r in records]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(filtered_mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(filtered_tags)))
    ax.set_xticklabels(filtered_tags, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(img_labels)))
    ax.set_yticklabels(img_labels, fontsize=9)
    ax.set_title(f"Batch Variable Heatmap: {slot_name}")
    fig.colorbar(im, ax=ax, label="WD14 Confidence")
    plt.tight_layout()
    plt.show()


def save_results_to_drive(output_dir: str, file_paths: list[str], prefix: str = ""):
    """生成した JSON / CSV / デバッグ画像を Google Drive へ保存する。"""
    if not os.path.exists("/content/drive"):
        print(
            "⚠️ 警告: '/content/drive' が見つかりません。先に Drive をマウントしてください。"
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    for src_path in file_paths:
        if not os.path.exists(src_path):
            print(f"ℹ️ スキップ（未生成）: {src_path}")
            continue
        dst_name = f"{prefix}{os.path.basename(src_path)}"
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        print(f"✅ 保存完了: {dst_path}")
        saved_count += 1
    print(f"合計 {saved_count} ファイルを Drive に保存しました -> {output_dir}")


def show_leaf_summary_chart(leaf_summary_df, top_n: int = 25, figsize=(12, 7)):
    """決定木の葉ノード別の採用数・除外数を積み上げ横棒グラフで可視化する。"""
    sub = leaf_summary_df.head(top_n).iloc[::-1]
    if len(sub) == 0:
        print("ℹ️ 表示する葉ノードがありません。")
        return

    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, sub["selected"], color="#54A24B", alpha=0.88, label="Selected")
    ax.barh(
        y,
        sub["rejected"],
        left=sub["selected"],
        color="#E45756",
        alpha=0.82,
        label="Rejected",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(sub.index.tolist(), fontsize=9)
    ax.set_xlabel("Image Count")
    ax.set_title(f"Taxonomy Leaf Distribution (Top {len(sub)} / {len(leaf_summary_df)} leaves)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def show_leaf_thumbnails(
    result_df,
    leaf_id: str | None = None,
    selected: bool | None = None,
    n: int = 12,
    cols: int = 4,
):
    """指定した葉ノード（または全体）の採用／除外画像サムネイルを一覧表示する。"""
    import math

    sub = result_df.copy()
    if leaf_id is not None:
        sub = sub[sub["leaf_id"] == leaf_id]
    if selected is not None:
        sub = sub[sub["selected"] == selected]
    sub = sub.head(n)

    if len(sub) == 0:
        print(f"ℹ️ 条件に該当する画像がありません (leaf_id={leaf_id}, selected={selected})。")
        return

    rows_n = math.ceil(len(sub) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.8, rows_n * 4.3))
    axes = np.atleast_1d(axes).ravel()
    title_parts = [leaf_id or "All Leaves"]
    if selected is not None:
        title_parts.append("Selected" if selected else "Rejected")
    fig.suptitle(f"{' | '.join(title_parts)} (showing {len(sub)})", fontsize=12)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sub.iterrows()):
        img = cv2.imread(row["file_path"])
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        is_sel = bool(row.get("selected", True))
        lines = [
            str(row["file_name"]),
            f"leaf={row['leaf_id']}",
            f"sim={row['leaf_centroid_sim']:.3f} | fr={row['top_framing_score']:.2f}",
        ]
        if not is_sel and row.get("reject_reasons"):
            lines.append(f"[{str(row['reject_reasons'])[:42]}]")
        ax.set_title("\n".join(lines), fontsize=8, color="#1B5E20" if is_sel else "#B71C1C")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def archive_folder_to_drive(
    src_dir: str,
    drive_output_dir: str,
    zip_name: str,
    local_work_dir: str = "./output",
    compress: bool = False,
):
    """フォルダ全体をローカル SSD 上で ZIP 化（デフォルト ZIP_STORED）し、Google Drive へコピーする。"""
    import zipfile
    from pathlib import Path

    if not os.path.isdir(src_dir):
        print(f"⚠️ 警告: 対象フォルダが見つかりません: {src_dir}")
        return None

    os.makedirs(local_work_dir, exist_ok=True)
    zip_file_name = zip_name if zip_name.endswith(".zip") else f"{zip_name}.zip"
    local_zip = os.path.join(local_work_dir, zip_file_name)
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    src_root = Path(src_dir).resolve()
    file_count = 0
    with zipfile.ZipFile(local_zip, "w", compression=mode) as zf:
        for p in sorted(src_root.rglob("*")):
            if p.is_file():
                arcname = Path(src_root.name) / p.relative_to(src_root)
                zf.write(p, arcname=str(arcname))
                file_count += 1

    size_mb = os.path.getsize(local_zip) / (1024 * 1024)
    print(f"📦 ZIP 作成完了: {local_zip} ({file_count} files, {size_mb:.1f} MB)")

    if not os.path.exists("/content/drive"):
        print("⚠️ 警告: '/content/drive' が見つかりません。ローカル ZIP のみ作成しました。")
        return local_zip

    os.makedirs(drive_output_dir, exist_ok=True)
    dst_zip = os.path.join(drive_output_dir, zip_file_name)
    shutil.copy2(local_zip, dst_zip)
    print(f"✅ Drive アップロード完了: {dst_zip}")
    return dst_zip
