"""スロットベースのプロンプト乱数生成ユーティリティ。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class PromptSlot:
    """単一プロンプト要素（framing / orientation / angle / emotion / posture 等）の乱数設定。

    Attributes
    ----------
    choices : Sequence[str] | Mapping[str, float]
        候補タグのリスト、または `{タグ: 重み}` の辞書。
    enabled : bool
        `True` のとき乱数生成を有効にする。`False` のときは乱数生成をオフにし、
        `fixed` が指定されていれば固定値を返し、未指定なら何も追加しない。
    probability : float
        `enabled=True` のときにこの要素を採用する確率（`0.0`〜`1.0`）。
        `0.0` の場合は実質オフと同じ動作になる。
    fixed : str | None
        `enabled=False` のときに代わりに使用する固定プロンプト（不要なら `None`）。
    """

    choices: Sequence[str] | Mapping[str, float] = field(default_factory=tuple)
    enabled: bool = True
    probability: float = 1.0
    fixed: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.probability, (int, float)) or isinstance(self.probability, bool):
            raise TypeError("probability は 0.0〜1.0 の数値で指定してください。")
        if not (0.0 <= float(self.probability) <= 1.0):
            raise ValueError(f"probability は 0.0〜1.0 の範囲で指定してください: {self.probability}")
        self.probability = float(self.probability)

        if isinstance(self.choices, str):
            raise TypeError("choices には文字列ではなく、文字列のリストまたは重み辞書を指定してください。")

        if isinstance(self.choices, Mapping):
            cleaned: dict[str, float] = {}
            for tag, weight in self.choices.items():
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise TypeError(f"候補 '{tag}' の重みは数値で指定してください: {weight}")
                if float(weight) < 0.0:
                    raise ValueError(f"候補 '{tag}' の重みは 0 以上で指定してください: {weight}")
                tag_str = str(tag).strip()
                if tag_str and float(weight) > 0.0:
                    cleaned[tag_str] = float(weight)
            self.choices = cleaned
        else:
            self.choices = tuple(str(c).strip() for c in self.choices if str(c).strip())

        if self.fixed is not None:
            fixed_str = str(self.fixed).strip()
            self.fixed = fixed_str or None

    @classmethod
    def from_config(cls, config: PromptSlot | Mapping[str, Any]) -> PromptSlot:
        """`PromptSlot` または設定辞書からインスタンスを構築する。"""
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError(f"スロット設定は PromptSlot または dict である必要があります: {type(config)}")
        return cls(
            choices=config.get("choices", ()),
            enabled=bool(config.get("enabled", True)),
            probability=float(config.get("probability", 1.0)),
            fixed=config.get("fixed"),
        )

    def sample(self, rng: random.Random | None = None) -> str | None:
        """設定に従ってタグを 1 件選択する（採用しない場合は `None`）。"""
        if not self.enabled:
            return self.fixed

        if self.probability <= 0.0 or not self.choices:
            return None

        random_gen = rng or random
        if self.probability < 1.0 and random_gen.random() >= self.probability:
            return None

        if isinstance(self.choices, Mapping):
            tags = list(self.choices.keys())
            weights = list(self.choices.values())
            if not tags:
                return None
            return random_gen.choices(tags, weights=weights, k=1)[0]

        return random_gen.choice(self.choices)


@dataclass(frozen=True)
class GeneratedPrompt:
    """1 サイクル分の生成プロンプトと各スロットの選択結果。"""

    positive: str
    negative: str
    selected_slots: dict[str, str | None] = field(default_factory=dict)


class PromptGenerator:
    """共通 positive/negative プロンプトに各要素の乱数タグを組み合わせて生成するクラス。"""

    def __init__(
        self,
        common_positive: str,
        common_negative: str = "",
        slots: Mapping[str, PromptSlot | Mapping[str, Any]] | None = None,
        *,
        separator: str = ", ",
        seed: int | None = None,
    ) -> None:
        self.common_positive = common_positive.strip()
        self.common_negative = common_negative.strip()
        self.separator = separator
        self.slots: dict[str, PromptSlot] = {
            name: PromptSlot.from_config(cfg) for name, cfg in (slots or {}).items()
        }
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def generate(self, rng: random.Random | None = None) -> GeneratedPrompt:
        """1 回分の positive / negative プロンプトを乱数生成する。"""
        random_gen = rng or self._rng
        selected: dict[str, str | None] = {}
        parts: list[str] = []

        if self.common_positive:
            parts.append(self.common_positive)

        for name, slot in self.slots.items():
            picked = slot.sample(random_gen)
            selected[name] = picked
            if picked:
                parts.append(picked)

        positive = self.separator.join(parts)
        return GeneratedPrompt(
            positive=positive,
            negative=self.common_negative,
            selected_slots=selected,
        )

    def generate_cycle(
        self,
        loop_count: int,
        *,
        rng: random.Random | None = None,
    ) -> list[GeneratedPrompt]:
        """指定したループ回数ぶん乱数プロンプトを生成してリストで返す。"""
        if not isinstance(loop_count, int) or isinstance(loop_count, bool) or loop_count < 1:
            raise ValueError(f"loop_count は 1 以上の整数で指定してください: {loop_count}")
        return [self.generate(rng=rng) for _ in range(loop_count)]
