"""inpaint_workflow.json (ComfyUI API format) を生成する。ComfyUI標準ノードのみ。
CKPT / LORA を自分の環境のファイル名に書き換えて実行。"""
import json, os
from dotenv import load_dotenv

load_dotenv()

COMFYUI_MODEL_PATH = os.environ["COMFYUI_MODEL_PATH"]
CHECKPOINT_SAFETENSORS = os.environ["CHECKPOINT_SAFETENSORS"]
LORA_SAFETENSORS = os.environ.get("LORA_SAFETENSORS")

CKPT = COMFYUI_MODEL_PATH + "/checkpoints/" + CHECKPOINT_SAFETENSORS
LORA = COMFYUI_MODEL_PATH + "/loras/" + LORA_SAFETENSORS \
    if LORA_SAFETENSORS else None

"""inpaint_workflow.json (ComfyUI API format) を生成する。ComfyUI標準ノードのみ。
CKPT / LORA を自分の環境のファイル名に書き換えて実行。"""
LORA_W = 0.4
POS = "flat small chest, loose pink tank top hanging from shoulders, soft vertical drape folds, same character"
NEG = "large breasts, tight clothes, wrinkles, deformed, lowres, bad anatomy"
wf = {
 "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":CKPT}},
 "2": {"class_type":"LoraLoader","inputs":{"model":["1",0],"clip":["1",1],"lora_name":LORA,
        "strength_model":LORA_W,"strength_clip":LORA_W}},
 "3": {"class_type":"CLIPTextEncode","inputs":{"clip":["2",1],"text":POS}},
 "4": {"class_type":"CLIPTextEncode","inputs":{"clip":["2",1],"text":NEG}},
 "5": {"class_type":"LoadImage","inputs":{"image":"target.png"}},
 "6": {"class_type":"LoadImageMask","inputs":{"image":"mask.png","channel":"red"}},
 "7": {"class_type":"GrowMask","inputs":{"mask":["6",0],"expand":4,"tapered_corners":True}},
 "8": {"class_type":"VAEEncode","inputs":{"pixels":["5",0],"vae":["1",2]}},
 "9": {"class_type":"SetLatentNoiseMask","inputs":{"samples":["8",0],"mask":["7",0]}},
 "10":{"class_type":"KSampler","inputs":{"model":["2",0],"positive":["3",0],"negative":["4",0],
        "latent_image":["9",0],"seed":1234,"steps":20,"cfg":6.5,"sampler_name":"euler_ancestral",
        "scheduler":"normal","denoise":0.30}},
 "11":{"class_type":"VAEDecode","inputs":{"samples":["10",0],"vae":["1",2]}},
 # マスク内だけ元画像に貼り戻す(顔・背景は完全に不変)
 "12":{"class_type":"ImageCompositeMasked","inputs":{"destination":["5",0],"source":["11",0],
        "x":0,"y":0,"resize_source":False,"mask":["7",0]}},
 "13":{"class_type":"SaveImage","inputs":{"images":["12",0],"filename_prefix":"chest_edit"}},
}
if LORA is None:
    del wf["2"]
    for n in wf.values():
        for k,v in n["inputs"].items():
            if v == ["2",0]: n["inputs"][k] = ["1",0]
            elif v == ["2",1]: n["inputs"][k] = ["1",1]
json.dump(wf, open("inpaint_workflow.json","w"), indent=1)
