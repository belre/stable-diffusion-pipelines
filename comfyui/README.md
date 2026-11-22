# ComfyUI ワークフロー実行環境 (`comfyui/`)

`workflows/` に **ComfyUI の API 形式ワークフロー JSON** を配置して画像生成を実行するための環境・パッケージです。  
Google Colab 上での完結実行（`colab_runner.ipynb`）、CLI 実行（`run.py`）、および将来の `colab/illustration-generator` 等からの Python モジュール呼び出し（`runner`）のすべてに対応しています。

## ディレクトリ構成

```text
comfyui/
├── colab_runner.ipynb      # Google Colab 完結ノートブック（SDXL ハイブリッド対応）
├── run.py                  # CLI エントリーポイント
├── config.example.yaml     # 設定ファイルサンプル（config.yaml にコピーして利用）
├── models.example.yaml     # 公開 SDXL モデル自動 DL 用マニフェストサンプル
├── requirements.txt        # オプション依存（WebSocket 進捗・YAML・Pillow）
├── presets/                # SDXL 等の生成パラメータプリセット
│   ├── sdxl_default.yaml   # SDXL 縦長（832x1216）デフォルトプリセット
│   └── sdxl_square.yaml    # SDXL 正方形（1024x1024）プリセット
├── runner/                 # 再利用可能な Python パッケージ
│   ├── client.py           # ComfyUI REST / WebSocket API クライアント
│   ├── workflow.py         # API 形式 JSON 読み込み・検証・SDXL プリセット＆上書き
│   ├── models.py           # 公開モデル自動 DL（HF/Civitai）＆参照モデル事前チェック
│   ├── server.py           # ComfyUI サーバー起動管理・Drive モデル連携
│   └── cli.py              # CLI 実装
├── workflows/              # 実行したい API 形式ワークフロー JSON を配置
├── input/                  # LoadImage 等で使う入力画像を配置（.gitignore 対象）
├── output/                 # 生成画像の保存先（.gitignore 対象）
└── tests/
    └── test_runner.py      # ユニットテスト
```

## SDXL モデル・パラメータ群の運用（ハイブリッド構成）

1. **公開 SDXL ベースモデル / VAE 等（大容量）**
   - `models.example.yaml` を参考に Google Drive 上の `MyDrive/ComfyUI/models.yaml`（またはノートブック内の `INLINE_MODELS`）へ記載すると、Colab 起動時に `aria2c` でローカル SSD（`/content/ComfyUI/models/`）へ高速ダウンロードされます。
   - 認証が必要なモデル（Gated な HF リポジトリや Civitai）は Colab Secrets に `HF_TOKEN` / `CIVITAI_API_KEY` を登録しておくと自動で付与されます。
2. **自作 LoRA・プライベートモデル**
   - Google Drive の `MyDrive/ComfyUI/models/loras/` 等に配置すると、`extra_model_paths.yaml` 経由でコピーなしに直接読み込まれます。
3. **SDXL 生成パラメータ群（プリセット）**
   - `presets/sdxl_default.yaml` 等で `checkpoint`, `vae`, `width`/`height`, `steps`, `cfg`, `sampler_name`, `scheduler`, `lora`, `negative_prompt` をまとめて管理できます。
   - `CLIPTextEncode` と `CLIPTextEncodeSDXL`（`text_g` / `text_l`・解像度メタデータ同期）の両方に対応しています。

## ワークフロー JSON の準備（API 形式）

ComfyUI の通常保存（UI 形式）ではなく、**API 形式**で書き出した JSON を使用します。

- **新しい ComfyUI UI**: メニュー `Workflow` → `Export (API)`
- **従来の ComfyUI UI**: 右上の歯車（Settings）→ `Enable Dev mode Options` を ON → `Save (API Format)`

## 使い方

### 1. Google Colab で完結実行する

1. `comfyui/colab_runner.ipynb` を Google Colab で開く（ランタイムは GPU を選択）。
2. 自作 LoRA 等は `MyDrive/ComfyUI/models/loras/` へ配置し、公開 SDXL モデルは `MyDrive/ComfyUI/models.yaml`（またはセル内の `INLINE_MODELS`）へ記載。
3. ノートブックのセルを上から実行すると、ComfyUI の導入・モデル準備・サーバー起動・参照モデル事前チェック・画像生成・Drive 保存まで自動で進みます。

### 2. CLI から実行する（起動済み ComfyUI サーバー向け）

```bash
# SDXL プリセットを適用して単一ワークフローを実行
python comfyui/run.py comfyui/workflows/sample_api.json \
  --preset comfyui/presets/sdxl_default.yaml \
  --positive "1girl, solo, anime style, masterpiece" \
  --random-seed

# workflows/ 内の全 JSON を一括実行
python comfyui/run.py --all --preset comfyui/presets/sdxl_default.yaml --random-seed
```

### 3. Python モジュールとして呼び出す（パイプライン連携向け）

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("comfyui").resolve()))
from runner import ComfyUIClient

client = ComfyUIClient("http://127.0.0.1:8188")
result = client.run_workflow(
    "comfyui/workflows/base_sdxl.json",
    output_dir="comfyui/output",
    preset="comfyui/presets/sdxl_default.yaml",
    positive_prompt="1girl, forest, sunlight",
    randomize_seed=True,
)
print(result.output_files)
```

## テストの実行

```bash
python3 -m unittest discover -s comfyui/tests -v
```
