"""comfyui/runner のユニットテスト（ComfyUI サーバー不要のモック検証）。"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# comfyui/ を sys.path に追加
COMFYUI_DIR = Path(__file__).resolve().parent.parent
if str(COMFYUI_DIR) not in sys.path:
    sys.path.insert(0, str(COMFYUI_DIR))

from runner import (  # noqa: E402
    ComfyUIClient,
    GeneratedPrompt,
    ModelSpec,
    PromptGenerator,
    PromptSlot,
    WorkflowFormatError,
    apply_overrides,
    apply_sdxl_preset,
    check_workflow_models,
    collect_input_image_nodes,
    download_models,
    find_nodes,
    generate_extra_model_paths_yaml,
    load_model_manifest,
    load_workflow,
    parse_override_spec,
    validate_api_workflow,
)
from runner.cli import main as cli_main  # noqa: E402


def make_sample_workflow() -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base_sdxl.safetensors"},
            "_meta": {"title": "Load Checkpoint"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "char_lora.safetensors",
                "strength_model": 0.7,
                "strength_clip": 0.7,
            },
            "_meta": {"title": "Load LoRA"},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "positive": ["6", 0],
                "negative": ["7", 0],
            },
            "_meta": {"title": "KSampler"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "_meta": {"title": "Empty Latent Image"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old positive"},
            "_meta": {"title": "Positive Prompt"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old negative"},
            "_meta": {"title": "Negative Prompt"},
        },
        "10": {
            "class_type": "LoadImage",
            "inputs": {"image": "ref.png"},
            "_meta": {"title": "Load Image"},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            "_meta": {"title": "Save Image"},
        },
    }


def make_sdxl_clip_workflow() -> dict:
    wf = make_sample_workflow()
    wf["6"] = {
        "class_type": "CLIPTextEncodeSDXL",
        "inputs": {
            "text_g": "old global",
            "text_l": "old local",
            "width": 1024,
            "height": 1024,
            "target_width": 1024,
            "target_height": 1024,
        },
        "_meta": {"title": "SDXL Positive"},
    }
    return wf


class TestWorkflowUtils(unittest.TestCase):
    def test_validate_rejects_ui_format(self) -> None:
        ui_workflow = {"last_node_id": 10, "nodes": [], "links": []}
        with self.assertRaises(WorkflowFormatError) as ctx:
            validate_api_workflow(ui_workflow)
        self.assertIn("API 形式", str(ctx.exception))

    def test_load_workflow_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wf.json"
            p.write_text(json.dumps(make_sample_workflow()), encoding="utf-8")
            loaded = load_workflow(p)
            self.assertIn("3", loaded)

    def test_find_nodes(self) -> None:
        wf = make_sample_workflow()
        self.assertEqual(find_nodes(wf, node_id="6"), ["6"])
        self.assertEqual(find_nodes(wf, title="positive prompt"), ["6"])
        self.assertEqual(find_nodes(wf, class_type="CLIPTextEncode"), ["6", "7"])

    def test_apply_overrides_all_modes(self) -> None:
        wf = make_sample_workflow()
        updated = apply_overrides(
            wf,
            overrides={
                "class:KSampler.steps": 30,
                "title:Save Image.filename_prefix": "custom",
                "3.cfg": 6.0,
            },
            positive_prompt="1girl, forest",
            negative_prompt="lowres",
            randomize_seed=True,
            rng=random.Random(12345),
        )
        # 元辞書が変更されていないこと
        self.assertEqual(wf["3"]["inputs"]["steps"], 20)
        self.assertEqual(wf["6"]["inputs"]["text"], "old positive")

        self.assertEqual(updated["3"]["inputs"]["steps"], 30)
        self.assertEqual(updated["3"]["inputs"]["cfg"], 6.0)
        self.assertNotEqual(updated["3"]["inputs"]["seed"], 42)
        self.assertEqual(updated["6"]["inputs"]["text"], "1girl, forest")
        self.assertEqual(updated["7"]["inputs"]["text"], "lowres")
        self.assertEqual(updated["9"]["inputs"]["filename_prefix"], "custom")

    def test_apply_sdxl_preset_and_clip_sdxl(self) -> None:
        wf = make_sdxl_clip_workflow()
        preset = {
            "checkpoint": "animagine-xl-3.1.safetensors",
            "width": 832,
            "height": 1216,
            "steps": 28,
            "cfg": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "karras",
            "lora": {
                "name": "my_char.safetensors",
                "strength_model": 0.85,
                "strength_clip": 0.8,
            },
            "positive_prompt": "1girl, solo, masterpiece",
            "negative_prompt": "lowres",
        }
        updated = apply_sdxl_preset(wf, preset)
        self.assertEqual(updated["1"]["inputs"]["ckpt_name"], "animagine-xl-3.1.safetensors")
        self.assertEqual(updated["2"]["inputs"]["lora_name"], "my_char.safetensors")
        self.assertEqual(updated["2"]["inputs"]["strength_model"], 0.85)
        self.assertEqual(updated["5"]["inputs"]["width"], 832)
        self.assertEqual(updated["5"]["inputs"]["height"], 1216)
        self.assertEqual(updated["3"]["inputs"]["steps"], 28)
        self.assertEqual(updated["3"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(updated["6"]["inputs"]["text_g"], "1girl, solo, masterpiece")
        self.assertEqual(updated["6"]["inputs"]["text_l"], "1girl, solo, masterpiece")
        self.assertEqual(updated["6"]["inputs"]["target_width"], 832)
        self.assertEqual(updated["6"]["inputs"]["target_height"], 1216)
        self.assertEqual(updated["7"]["inputs"]["text"], "lowres")

    def test_apply_fixed_seed_takes_precedence(self) -> None:
        wf = make_sample_workflow()
        updated = apply_overrides(wf, randomize_seed=True, seed=9999)
        self.assertEqual(updated["3"]["inputs"]["seed"], 9999)

    def test_parse_override_spec(self) -> None:
        self.assertEqual(parse_override_spec("class:KSampler.steps=28"), ("class:KSampler.steps", 28))
        self.assertEqual(parse_override_spec("3.cfg=6.5"), ("3.cfg", 6.5))
        self.assertEqual(parse_override_spec("6.text=1girl, solo"), ("6.text", "1girl, solo"))

    def test_collect_input_image_nodes(self) -> None:
        wf = make_sample_workflow()
        self.assertEqual(collect_input_image_nodes(wf), [("10", "ref.png")])


class TestServerAndModelHelpers(unittest.TestCase):
    def test_generate_extra_model_paths_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "extra_model_paths.yaml"
            generated = generate_extra_model_paths_yaml("/content/drive/MyDrive/ComfyUI/models", out)
            self.assertEqual(generated, out)
            content = out.read_text(encoding="utf-8")
            self.assertIn("base_path: /content/drive/MyDrive/ComfyUI/models", content)
            self.assertIn("checkpoints: checkpoints/", content)
            self.assertIn("loras: loras/", content)

    def test_model_spec_resolve_url_and_manifest(self) -> None:
        spec = ModelSpec(
            subdir="checkpoints",
            filename="animagine-xl-3.1.safetensors",
            hf_repo="cagliostrolab/animagine-xl-3.1",
            hf_file="animagine-xl-3.1.safetensors",
        )
        self.assertEqual(
            spec.resolve_url(),
            "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors",
        )
        self.assertTrue(spec.is_huggingface)

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "subdir": "checkpoints",
                                "filename": "a.safetensors",
                                "url": "https://example.com/a.safetensors",
                                "enabled": True,
                            },
                            {
                                "subdir": "checkpoints",
                                "filename": "disabled.safetensors",
                                "url": "https://example.com/b.safetensors",
                                "enabled": False,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_model_manifest(manifest)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].filename, "a.safetensors")

    def test_check_workflow_models_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            comfy_dir = tmp_path / "ComfyUI"
            drive_models = tmp_path / "DriveModels"
            (comfy_dir / "models" / "checkpoints").mkdir(parents=True)
            (drive_models / "loras").mkdir(parents=True)

            # Checkpoint はローカル、LoRA は Drive にあるハイブリッド状態
            (comfy_dir / "models" / "checkpoints" / "base_sdxl.safetensors").write_bytes(b"ckpt")
            (drive_models / "loras" / "char_lora.safetensors").write_bytes(b"lora")

            result = check_workflow_models(
                make_sample_workflow(),
                comfyui_dir=comfy_dir,
                drive_models_dir=drive_models,
            )
            self.assertEqual(result["missing"], [])
            self.assertEqual(
                set(result["found"]),
                {"checkpoints/base_sdxl.safetensors", "loras/char_lora.safetensors"},
            )

    def test_download_models_skips_drive_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            comfy_dir = tmp_path / "ComfyUI"
            drive_models = tmp_path / "DriveModels"
            (drive_models / "loras").mkdir(parents=True)
            existing = drive_models / "loras" / "char_lora.safetensors"
            existing.write_bytes(b"existing")

            specs = [
                ModelSpec(
                    subdir="loras",
                    filename="char_lora.safetensors",
                    url="https://example.com/char_lora.safetensors",
                )
            ]
            with mock.patch("runner.models._download_file") as m_dl:
                paths = download_models(specs, comfyui_dir=comfy_dir, drive_models_dir=drive_models)
                m_dl.assert_not_called()
                self.assertEqual(paths, [existing])

    def test_download_file_aria2_exit_24_raises_helpful_error(self) -> None:
        from runner.models import _download_file

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "anikawaxl_v4.safetensors"
            fake_proc = mock.Mock(returncode=24)
            with (
                mock.patch("runner.models.shutil.which", return_value="/usr/bin/aria2c"),
                mock.patch("runner.models.subprocess.run", return_value=fake_proc),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _download_file("https://civitai.com/api/download/models/2870022", dest)
                self.assertIn("CIVITAI_API_KEY", str(ctx.exception))


class TestClientAndCLI(unittest.TestCase):
    def test_run_workflow_with_mocked_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wf_path = tmp_path / "sample.json"
            wf_path.write_text(json.dumps(make_sample_workflow()), encoding="utf-8")

            input_dir = tmp_path / "input"
            input_dir.mkdir()
            (input_dir / "ref.png").write_bytes(b"FAKEPNG")
            output_dir = tmp_path / "output"

            client = ComfyUIClient("http://127.0.0.1:8188")

            with (
                mock.patch.object(client, "upload_image", return_value={"name": "ref.png", "subfolder": ""}) as m_upload,
                mock.patch.object(client, "queue_prompt", return_value="pid-123") as m_queue,
                mock.patch.object(
                    client,
                    "wait_for_completion",
                    return_value={
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "out_00001_.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    },
                ),
                mock.patch.object(client, "download_file", return_value=b"GENERATED_IMAGE"),
            ):
                result = client.run_workflow(
                    wf_path,
                    output_dir=output_dir,
                    input_dir=input_dir,
                    preset={"width": 832, "height": 1216, "steps": 28},
                    positive_prompt="masterpiece",
                    seed=777,
                    use_websocket=False,
                )

            m_upload.assert_called_once()
            queued_wf = m_queue.call_args.args[0]
            self.assertEqual(queued_wf["5"]["inputs"]["width"], 832)
            self.assertEqual(queued_wf["3"]["inputs"]["steps"], 28)
            self.assertEqual(queued_wf["6"]["inputs"]["text"], "masterpiece")
            self.assertEqual(queued_wf["3"]["inputs"]["seed"], 777)
            self.assertEqual(len(result.output_files), 1)
            self.assertEqual(result.output_files[0].read_bytes(), b"GENERATED_IMAGE")

    def test_cli_runs_single_workflow_with_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wf_path = tmp_path / "demo.json"
            wf_path.write_text(json.dumps(make_sample_workflow()), encoding="utf-8")
            preset_path = tmp_path / "preset.json"
            preset_path.write_text(json.dumps({"width": 832, "height": 1216}), encoding="utf-8")
            out_dir = tmp_path / "out"

            with (
                mock.patch.object(ComfyUIClient, "is_alive", return_value=True),
                mock.patch.object(ComfyUIClient, "queue_prompt", return_value="pid-cli") as m_queue,
                mock.patch.object(
                    ComfyUIClient,
                    "wait_for_completion",
                    return_value={
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "cli_out.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    },
                ),
                mock.patch.object(ComfyUIClient, "download_file", return_value=b"CLI_IMG"),
            ):
                code = cli_main(
                    [
                        str(wf_path),
                        "--output",
                        str(out_dir),
                        "--preset",
                        str(preset_path),
                        "--positive",
                        "1girl",
                        "--seed",
                        "123",
                        "--no-ws",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(m_queue.call_args.args[0]["5"]["inputs"]["width"], 832)
            self.assertTrue((out_dir / "cli_out.png").is_file())


class TestPromptGenerator(unittest.TestCase):
    def test_generate_with_five_slots_and_probabilities(self) -> None:
        slots = {
            "framing": {"enabled": True, "probability": 1.0, "choices": ["full body", "cowboy shot"]},
            "orientation": {"enabled": True, "probability": 1.0, "choices": {"looking at viewer": 0.8, "profile": 0.2}},
            "angle": {"enabled": True, "probability": 0.0, "choices": ["from above", "from below"]},
            "emotion": {"enabled": False, "probability": 1.0, "choices": ["smile", "angry"]},
            "posture": {"enabled": False, "probability": 1.0, "choices": ["standing", "sitting"], "fixed": "standing"},
        }
        gen = PromptGenerator(
            common_positive="maniol-elvish, 1girl, solo",
            common_negative="lowres, bad anatomy",
            slots=slots,
            seed=42,
        )
        result = gen.generate()
        self.assertIsInstance(result, GeneratedPrompt)
        self.assertEqual(result.negative, "lowres, bad anatomy")
        self.assertIn(result.selected_slots["framing"], {"full body", "cowboy shot"})
        self.assertIn(result.selected_slots["orientation"], {"looking at viewer", "profile"})
        self.assertIsNone(result.selected_slots["angle"])
        self.assertIsNone(result.selected_slots["emotion"])
        self.assertEqual(result.selected_slots["posture"], "standing")
        self.assertTrue(result.positive.startswith("maniol-elvish, 1girl, solo, "))
        self.assertIn("standing", result.positive)

    def test_generate_cycle_count_and_reproducibility(self) -> None:
        slots = {
            "framing": PromptSlot(choices=["portrait", "upper body", "full body"], probability=1.0),
            "emotion": PromptSlot(choices=["smile", "serious", "blush"], probability=0.5),
        }
        gen_a = PromptGenerator("base", "neg", slots=slots, seed=123)
        gen_b = PromptGenerator("base", "neg", slots=slots, seed=123)

        cycle_a = gen_a.generate_cycle(5)
        cycle_b = gen_b.generate_cycle(5)

        self.assertEqual(len(cycle_a), 5)
        self.assertEqual(cycle_a, cycle_b)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            PromptSlot(choices=["full body"], probability=1.5)
        with self.assertRaises(ValueError):
            PromptSlot(choices=["full body"], probability=-0.1)
        with self.assertRaises(TypeError):
            PromptSlot(choices="full body")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PromptSlot(choices={"full body": -1.0})

        gen = PromptGenerator("base")
        with self.assertRaises(ValueError):
            gen.generate_cycle(0)


if __name__ == "__main__":
    unittest.main()
