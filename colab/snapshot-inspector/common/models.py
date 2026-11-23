"""Model initialization helpers for rtmlib Wholebody and WD14-Tagger (ONNX)."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from rtmlib import Wholebody

DEFAULT_WD14_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"


def get_onnx_device_config() -> tuple[str, list[str]]:
    """Return (rtmlib_device, ort_providers) based on available ONNX Runtime providers."""
    available = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available
    rtmlib_device = "cuda" if use_cuda else "cpu"
    ort_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    return rtmlib_device, ort_providers


def init_wholebody(
    mode: str = "performance",
    to_openpose: bool = False,
    device: str | None = None,
) -> Wholebody:
    """Initialize rtmlib Wholebody (COCO WholeBody 133 keypoints) model."""
    if device is None:
        device, _ = get_onnx_device_config()
    return Wholebody(
        to_openpose=to_openpose,
        mode=mode,
        backend="onnxruntime",
        device=device,
    )


class WD14Tagger:
    """
    SmilingWolf 系 WD14-Tagger (ONNX) を onnxruntime で直接推論するラッパー。
    タグの連続値信頼度スコア（0.0〜1.0）を取得できる。
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_WD14_REPO,
        providers: list[str] | None = None,
    ):
        self.repo_id = repo_id
        if providers is None:
            _, providers = get_onnx_device_config()
        self.providers = providers

        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
        tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")

        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        # 入力形状 [batch, height, width, 3] から正方形リサイズ解像度（通常448）を自動取得
        self.target_size = int(self.input_meta.shape[1])

        tags_df = pd.read_csv(tags_path)
        self.tag_names = tags_df["name"].astype(str).tolist()
        self.tag_categories = tags_df["category"].astype(int).tolist()
        self.rating_indices = [i for i, c in enumerate(self.tag_categories) if c == 9]
        self.general_indices = [i for i, c in enumerate(self.tag_categories) if c == 0]
        self.character_indices = [i for i, c in enumerate(self.tag_categories) if c == 4]

        print(
            f"Loaded WD14Tagger: {repo_id} "
            f"(size={self.target_size}, tags={len(self.tag_names)})"
        )

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        WD14-Tagger v3 標準の前処理:
        白背景で正方形パディング -> target_size へリサイズ -> BGR float32 (NHWC)。
        """
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img_bgr[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 255.0)
            img_bgr = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)

        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        padded = cv2.copyMakeBorder(
            img_bgr,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        interp = cv2.INTER_AREA if max_dim > self.target_size else cv2.INTER_CUBIC
        resized = cv2.resize(
            padded, (self.target_size, self.target_size), interpolation=interp
        )
        # SmilingWolf ONNX モデルは BGR 順・[0, 255] スケールの float32 を期待する
        tensor = resized.astype(np.float32)
        return np.expand_dims(tensor, axis=0)

    def predict(self, img_bgr: np.ndarray) -> dict:
        """
        画像を推論し、rating / general / character の全信頼度スコア辞書を返す。
        """
        tensor = self.preprocess(img_bgr)
        probs = self.session.run([self.output_name], {self.input_name: tensor})[0][0]

        return {
            "rating": {self.tag_names[i]: float(probs[i]) for i in self.rating_indices},
            "general": {
                self.tag_names[i]: float(probs[i]) for i in self.general_indices
            },
            "character": {
                self.tag_names[i]: float(probs[i]) for i in self.character_indices
            },
        }
