"""Pose variable extraction from COCO WholeBody (133 keypoints) detections."""

from __future__ import annotations

import math

import numpy as np

# COCO Body 17キーポイントのインデックス定義（WholeBody 133点の先頭17点と一致）
COCO_BODY_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KP = {name: idx for idx, name in enumerate(COCO_BODY_NAMES)}


def round_or_none(val, digits: int = 4):
    if val is None or (
        isinstance(val, float) and (math.isnan(val) or math.isinf(val))
    ):
        return None
    return round(float(val), digits)


def _select_primary_person(kpts, scores, kpt_thr: float = 0.3):
    """複数人検出された場合、体幹・顔を含む有効点が多く面積が大きいメイン人物を選択する。"""
    if kpts is None or len(kpts) == 0:
        return None, None
    if len(kpts) == 1:
        return (
            np.asarray(kpts[0], dtype=np.float32),
            np.asarray(scores[0], dtype=np.float32),
        )

    best_idx = 0
    best_weight = -1.0
    for idx in range(len(kpts)):
        body_pts = np.asarray(kpts[idx][:17], dtype=np.float32)
        body_sc = np.asarray(scores[idx][:17], dtype=np.float32)
        valid = body_sc >= kpt_thr
        if not np.any(valid):
            continue
        pts = body_pts[valid]
        area = float(
            (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        )
        weight = float(body_sc[valid].mean()) * (area**0.5 + 1.0)
        if weight > best_weight:
            best_weight = weight
            best_idx = idx
    return (
        np.asarray(kpts[best_idx], dtype=np.float32),
        np.asarray(scores[best_idx], dtype=np.float32),
    )


def _point_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """3点 A-B-C における頂点 B の内角（0〜180度）を計算する。"""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_val = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def _line_tilt_deg(p_left: np.ndarray, p_right: np.ndarray):
    """画面左側の点から右側の点へのラインの水平からの傾き（度）を計算する。"""
    dx = float(p_left[0] - p_right[0])
    dy = float(p_left[1] - p_right[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(dy, dx)))


def extract_pose_variables(kpts, scores, img_shape, kpt_thr: float = 0.35) -> dict:
    """
    rtmlib WholeBody の検出結果から、立ち絵・アニメーション向けの数値変数を抽出する。
    """
    h, w = img_shape[:2]
    p, s = _select_primary_person(kpts, scores, kpt_thr=kpt_thr)

    if p is None or s is None:
        return {
            "detection_quality": {
                "is_valid": False,
                "person_count": 0,
                "body_valid_ratio": 0.0,
                "body_mean_score": 0.0,
            },
            "framing": {},
            "orientation": {},
            "proportions": {},
            "joint_angles": {},
            "normalized_keypoints": {},
        }

    body_p = p[:17]
    body_s = s[:17]

    def has(*names: str) -> bool:
        return all(body_s[KP[n]] >= kpt_thr for n in names)

    def pt(name: str) -> np.ndarray:
        return body_p[KP[name]]

    def in_bounds(name: str) -> bool:
        if not has(name):
            return False
        x, y = float(pt(name)[0]), float(pt(name)[1])
        return (0.0 <= x < w) and (0.0 <= y < h)

    valid_mask = body_s >= kpt_thr
    valid_pts = body_p[valid_mask]

    # --- 1. 画角・構図（framing） ---
    bbox_w_ratio = bbox_h_ratio = center_x = center_y = None
    if len(valid_pts) > 0:
        clipped_x = np.clip(valid_pts[:, 0], 0.0, float(w))
        clipped_y = np.clip(valid_pts[:, 1], 0.0, float(h))
        bbox_w_ratio = (clipped_x.max() - clipped_x.min()) / w
        bbox_h_ratio = (clipped_y.max() - clipped_y.min()) / h

    if has("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        core = np.mean(
            [
                pt("left_shoulder"),
                pt("right_shoulder"),
                pt("left_hip"),
                pt("right_hip"),
            ],
            axis=0,
        )
        center_x, center_y = core[0] / w, core[1] / h
    elif has("left_shoulder", "right_shoulder"):
        core = np.mean([pt("left_shoulder"), pt("right_shoulder")], axis=0)
        center_x, center_y = core[0] / w, core[1] / h
    elif len(valid_pts) > 0:
        core = np.mean(valid_pts, axis=0)
        center_x, center_y = core[0] / w, core[1] / h

    shoulder_width = (
        float(np.linalg.norm(pt("left_shoulder") - pt("right_shoulder")))
        if has("left_shoulder", "right_shoulder")
        else None
    )
    eye_dist = (
        float(np.linalg.norm(pt("left_eye") - pt("right_eye")))
        if has("left_eye", "right_eye")
        else None
    )
    shoulder_width_ratio = (shoulder_width / w) if shoulder_width is not None else None
    eye_dist_ratio = (eye_dist / w) if eye_dist is not None else None

    # 画像枠内に実際に収まっている最下部部位からショット種別を推定（画面外予測による誤判定を防止）
    has_ankles = in_bounds("left_ankle") or in_bounds("right_ankle")
    has_knees = in_bounds("left_knee") or in_bounds("right_knee")
    has_hips = in_bounds("left_hip") or in_bounds("right_hip")
    has_shoulders = in_bounds("left_shoulder") or in_bounds("right_shoulder")
    if has_ankles:
        estimated_shot = "full_body"
    elif has_knees:
        estimated_shot = "cowboy_shot"
    elif has_hips:
        estimated_shot = "upper_body"
    elif has_shoulders:
        estimated_shot = "bust_up"
    else:
        estimated_shot = "close_up"

    # --- 2. 向き・傾き（orientation） ---
    head_roll_deg = (
        _line_tilt_deg(pt("left_eye"), pt("right_eye"))
        if has("left_eye", "right_eye")
        else None
    )
    shoulder_tilt_deg = (
        _line_tilt_deg(pt("left_shoulder"), pt("right_shoulder"))
        if has("left_shoulder", "right_shoulder")
        else None
    )

    # 体の前後向き判定（左肩・左腰が画面右にあれば前向き、画面左に反転していれば後ろ向き）
    side_eps = max(w * 0.02, 5.0)
    shoulder_facing = None
    if has("left_shoulder", "right_shoulder"):
        dx_sh = float(pt("left_shoulder")[0] - pt("right_shoulder")[0])
        if abs(dx_sh) < side_eps:
            shoulder_facing = "side"
        else:
            shoulder_facing = "front" if dx_sh > 0 else "back"

    hip_facing = None
    if has("left_hip", "right_hip"):
        dx_hip = float(pt("left_hip")[0] - pt("right_hip")[0])
        if abs(dx_hip) < side_eps:
            hip_facing = "side"
        else:
            hip_facing = "front" if dx_hip > 0 else "back"

    if shoulder_facing and hip_facing:
        if shoulder_facing == hip_facing:
            body_facing = shoulder_facing
        elif shoulder_facing == "side":
            body_facing = hip_facing
        elif hip_facing == "side":
            body_facing = shoulder_facing
        else:
            body_facing = "twisted"
    else:
        body_facing = shoulder_facing or hip_facing or "unknown"

    # 見返りポーズ判定（顔は前向きなのに、肩または体幹が後ろ向きに反転している状態）
    is_looking_back_pose = bool(
        head_roll_deg is not None
        and abs(head_roll_deg) < 90.0
        and (body_facing == "back" or shoulder_facing == "back")
    )

    head_yaw_proxy = None
    if has("nose", "left_eye", "right_eye") and eye_dist and eye_dist > 1e-3:
        eye_mid = 0.5 * (pt("left_eye") + pt("right_eye"))
        eye_vec = pt("left_eye") - pt("right_eye")
        eye_unit = eye_vec / np.linalg.norm(eye_vec)
        nose_offset = float(np.dot(pt("nose") - eye_mid, eye_unit))
        head_yaw_proxy = float(np.clip(nose_offset / (eye_dist * 0.5), -2.0, 2.0))

    torso_tilt_deg = None
    torso_len = None
    if has("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        sh_mid = 0.5 * (pt("left_shoulder") + pt("right_shoulder"))
        hip_mid = 0.5 * (pt("left_hip") + pt("right_hip"))
        torso_vec = hip_mid - sh_mid
        torso_len = float(np.linalg.norm(torso_vec))
        if torso_len > 1e-3:
            torso_tilt_deg = float(np.degrees(np.arctan2(torso_vec[0], torso_vec[1])))

    if head_yaw_proxy is None:
        facing_direction = "unknown"
    elif abs(head_yaw_proxy) < 0.25:
        facing_direction = "front"
    elif abs(head_yaw_proxy) < 0.75:
        facing_direction = (
            "three_quarter_right" if head_yaw_proxy > 0 else "three_quarter_left"
        )
    else:
        facing_direction = "side_right" if head_yaw_proxy > 0 else "side_left"

    # --- 3. スケール不変プロポーション（proportions） ---
    eye_to_shoulder_ratio = None
    torso_to_shoulder_ratio = None
    hip_to_shoulder_ratio = None
    upper_arm_to_shoulder_ratio = None
    if shoulder_width and shoulder_width > 1e-3:
        if eye_dist is not None:
            eye_to_shoulder_ratio = eye_dist / shoulder_width
        if torso_len is not None:
            torso_to_shoulder_ratio = torso_len / shoulder_width
        if has("left_hip", "right_hip"):
            hip_width = float(np.linalg.norm(pt("left_hip") - pt("right_hip")))
            hip_to_shoulder_ratio = hip_width / shoulder_width
        arm_lens = []
        if has("left_shoulder", "left_elbow"):
            arm_lens.append(float(np.linalg.norm(pt("left_elbow") - pt("left_shoulder"))))
        if has("right_shoulder", "right_elbow"):
            arm_lens.append(
                float(np.linalg.norm(pt("right_elbow") - pt("right_shoulder")))
            )
        if arm_lens:
            upper_arm_to_shoulder_ratio = float(np.mean(arm_lens)) / shoulder_width

    # --- 4. 関節角度（joint_angles） ---
    left_elbow_deg = (
        _point_angle_deg(pt("left_shoulder"), pt("left_elbow"), pt("left_wrist"))
        if has("left_shoulder", "left_elbow", "left_wrist")
        else None
    )
    right_elbow_deg = (
        _point_angle_deg(pt("right_shoulder"), pt("right_elbow"), pt("right_wrist"))
        if has("right_shoulder", "right_elbow", "right_wrist")
        else None
    )
    left_knee_deg = (
        _point_angle_deg(pt("left_hip"), pt("left_knee"), pt("left_ankle"))
        if has("left_hip", "left_knee", "left_ankle")
        else None
    )
    right_knee_deg = (
        _point_angle_deg(pt("right_hip"), pt("right_knee"), pt("right_ankle"))
        if has("right_hip", "right_knee", "right_ankle")
        else None
    )

    down_ref = np.array([0.0, 1.0], dtype=np.float32)
    if (
        has("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        and torso_len
        and torso_len > 1e-3
    ):
        down_ref = (
            0.5 * (pt("left_hip") + pt("right_hip"))
            - 0.5 * (pt("left_shoulder") + pt("right_shoulder"))
        ) / torso_len

    def _arm_elev(sh_name: str, el_name: str):
        if not has(sh_name, el_name):
            return None
        v = pt(el_name) - pt(sh_name)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        return float(np.degrees(np.arccos(np.clip(np.dot(v / n, down_ref), -1.0, 1.0))))

    left_shoulder_elev_deg = _arm_elev("left_shoulder", "left_elbow")
    right_shoulder_elev_deg = _arm_elev("right_shoulder", "right_elbow")

    # --- 5. 正規化キーポイント（肩中点原点・肩幅=1.0） ---
    normalized_keypoints = {}
    if has("left_shoulder", "right_shoulder") and shoulder_width and shoulder_width > 1e-3:
        origin = 0.5 * (pt("left_shoulder") + pt("right_shoulder"))
        for idx, name in enumerate(COCO_BODY_NAMES):
            rel = (body_p[idx] - origin) / shoulder_width
            normalized_keypoints[name] = {
                "x": round_or_none(rel[0], 4),
                "y": round_or_none(rel[1], 4),
                "score": round_or_none(body_s[idx], 4),
            }

    return {
        "detection_quality": {
            "is_valid": bool(np.sum(valid_mask) >= 4),
            "person_count": int(len(kpts)),
            "body_valid_ratio": round_or_none(float(np.mean(valid_mask)), 4),
            "body_mean_score": round_or_none(float(np.mean(body_s)), 4),
        },
        "framing": {
            "estimated_shot": estimated_shot,
            "bbox_w_ratio": round_or_none(bbox_w_ratio),
            "bbox_h_ratio": round_or_none(bbox_h_ratio),
            "center_x": round_or_none(center_x),
            "center_y": round_or_none(center_y),
            "shoulder_width_ratio": round_or_none(shoulder_width_ratio),
            "eye_dist_ratio": round_or_none(eye_dist_ratio),
        },
        "orientation": {
            "facing_direction": facing_direction,
            "body_facing": body_facing,
            "is_looking_back_pose": is_looking_back_pose,
            "head_yaw_proxy": round_or_none(head_yaw_proxy),
            "head_roll_deg": round_or_none(head_roll_deg, 2),
            "shoulder_tilt_deg": round_or_none(shoulder_tilt_deg, 2),
            "torso_tilt_deg": round_or_none(torso_tilt_deg, 2),
        },
        "proportions": {
            "eye_to_shoulder_ratio": round_or_none(eye_to_shoulder_ratio),
            "torso_to_shoulder_ratio": round_or_none(torso_to_shoulder_ratio),
            "hip_to_shoulder_ratio": round_or_none(hip_to_shoulder_ratio),
            "upper_arm_to_shoulder_ratio": round_or_none(upper_arm_to_shoulder_ratio),
        },
        "joint_angles": {
            "left_elbow_deg": round_or_none(left_elbow_deg, 2),
            "right_elbow_deg": round_or_none(right_elbow_deg, 2),
            "left_shoulder_elev_deg": round_or_none(left_shoulder_elev_deg, 2),
            "right_shoulder_elev_deg": round_or_none(right_shoulder_elev_deg, 2),
            "left_knee_deg": round_or_none(left_knee_deg, 2),
            "right_knee_deg": round_or_none(right_knee_deg, 2),
        },
        "normalized_keypoints": normalized_keypoints,
    }
