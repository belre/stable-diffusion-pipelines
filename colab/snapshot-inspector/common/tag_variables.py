"""WD14-Tagger slot definitions and structured tag variable extraction."""

from __future__ import annotations

DEFAULT_VARIABLE_TAG_SLOTS: dict[str, list[str]] = {
    "expression": [
        "smile",
        "grin",
        "open_mouth",
        "closed_mouth",
        "parted_lips",
        "blush",
        "expressionless",
        "serious",
        "angry",
        "sad",
        "surprised",
        "frown",
        "embarrassed",
        "nervous",
        "tears",
        "closed_eyes",
        "one_eye_closed",
        "half-closed_eyes",
        "wide-eyed",
    ],
    "gaze": [
        "looking_at_viewer",
        "looking_away",
        "looking_to_the_side",
        "looking_back",
        "looking_up",
        "looking_down",
        "facing_viewer",
        "facing_away",
        "profile",
        "from_side",
        "from_behind",
        "straight-on",
    ],
    "framing": [
        "portrait",
        "close-up",
        "upper_body",
        "cowboy_shot",
        "full_body",
        "wide_shot",
        "dutch_angle",
        "from_above",
        "from_below",
    ],
    "pose_action": [
        "standing",
        "sitting",
        "walking",
        "running",
        "jumping",
        "arms_up",
        "arm_up",
        "hand_on_hip",
        "hands_on_hips",
        "crossed_arms",
        "peace_sign",
        "waving",
        "holding",
        "leaning_forward",
        "head_tilt",
        "hands_behind_back",
        "arms_behind_back",
    ],
    "background": [
        "simple_background",
        "white_background",
        "grey_background",
        "black_background",
        "blue_background",
        "gradient_background",
        "transparent_background",
        "outdoors",
        "indoors",
        "sky",
    ],
}

DEFAULT_IGNORED_IDENTITY_TAGS: set[str] = {
    "1girl",
    "1boy",
    "solo",
    "solo_focus",
    "2girls",
    "2boys",
    "multiple_girls",
    "monochrome",
    "greyscale",
    "comic",
    "highres",
    "absurdres",
    "commentary_request",
    "shadow",
    "depth_of_field",
    "blurry",
    "blurry_background",
    "motion_blur",
    "light_rays",
    "lens_flare",
    "vignetting",
}


def extract_tag_variables(
    wd14_pred: dict,
    slot_definitions: dict[str, list[str]] | None = None,
    general_threshold: float = 0.35,
    character_threshold: float = 0.50,
) -> dict:
    """
    WD14-Tagger の推論結果から、カテゴリ別の連続値スロット変数と高信頼度タグを抽出する。
    """
    slots_cfg = (
        slot_definitions
        if slot_definitions is not None
        else DEFAULT_VARIABLE_TAG_SLOTS
    )
    general_scores = wd14_pred["general"]
    rating_scores = wd14_pred["rating"]
    character_scores = wd14_pred["character"]

    slots = {}
    slot_dominant = {}
    for slot_name, tag_list in slots_cfg.items():
        slot_dict = {
            tag: round(float(general_scores.get(tag, 0.0)), 4) for tag in tag_list
        }
        slots[slot_name] = slot_dict
        if slot_dict:
            top_tag, top_score = max(slot_dict.items(), key=lambda kv: kv[1])
            slot_dominant[slot_name] = {"tag": top_tag, "score": top_score}

    active_general = {
        tag: round(float(score), 4)
        for tag, score in sorted(
            general_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        if score >= general_threshold
    }
    active_character = {
        tag: round(float(score), 4)
        for tag, score in sorted(
            character_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        if score >= character_threshold
    }

    return {
        "slots": slots,
        "slot_dominant": slot_dominant,
        "rating": {k: round(float(v), 4) for k, v in rating_scores.items()},
        "active_general_tags": active_general,
        "active_character_tags": active_character,
    }
