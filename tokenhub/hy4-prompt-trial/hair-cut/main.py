import base64, os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["TOKENHUB_API_KEY"],
    base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",  # コンソール表示のURLを優先
    timeout=180,
)

def img(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

with open("prompt.txt", encoding="utf-8") as f:
    PROMPT = f.read()  # 先ほどの英語プロンプト

stream = client.chat.completions.create(
    model="hy4",
    messages=[
        {"role": "system", "content": "Always reply in English. Do not generate images."},
        {"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "text", "text": "Reference image:"},
            img("../../../local/maniol-anikawa-current_p2.png"),
            {"type": "text", "text": "Target image:"},
            img("../../../local/SAJ6O4sj.png"),
        ]},
    ],
    max_tokens=32000,
    stream=True,
)

parts = []
finish_reason = None

messages=[
    {"role": "system", "content": "Always reply in English. Do not generate images."},
    {"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "text", "text": "Original image:"},
        img("../../../local/sera-elvish-v5-mylora-b_00200_.png"),
        {"type": "text", "text": "Original with mask regions shown as pale glowing blobs image:"},
        img("../../../local/sera-elvish-v5-mylora-b_00200-wrapped.png"),
        {"type": "text", "text": "Mask image:"},
        img("../../../local/sera-elvish-v5-mylora-b_00200_mask.png"),
    ]},
]



# ---- 設定(ここだけ変える) ----
MAX_TOKENS = 32000
REASONING_EFFORT = "medium"     # "low" / "medium" / "high" / None(指定しない)
SHOW_REASONING = False       # True: 思考も画面に表示する
SAVE_REASONING = True        # True: 思考を別ファイルに保存する
# --------------------------------

kwargs = dict(
    model="hy4",
    messages=messages,       # 前回と同じ messages
    max_tokens=MAX_TOKENS,
    stream=True,
)
if REASONING_EFFORT:
    kwargs["reasoning_effort"] = REASONING_EFFORT

stream = client.chat.completions.create(**kwargs)

content_parts, reasoning_parts = [], []
finish_reason = None

for chunk in stream:
    if not chunk.choices:
        continue
    choice = chunk.choices[0]
    delta = choice.delta

    r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    if r:
        reasoning_parts.append(r)
        if SHOW_REASONING:
            print(r, end="", flush=True)
        else:
            print(".", end="", flush=True)   # 進捗の印だけ

    if delta.content:
        content_parts.append(delta.content)
        print(delta.content, end="", flush=True)

    if choice.finish_reason:
        finish_reason = choice.finish_reason

print()
content = "".join(content_parts)
reasoning = "".join(reasoning_parts)
print(f"[finish={finish_reason}, reasoning_chars={len(reasoning)}, content_chars={len(content)}]")

with open("hy4_eval.txt", "w", encoding="utf-8") as f:
    f.write(content)
if SAVE_REASONING:
    with open("hy4_reasoning.txt", "w", encoding="utf-8") as f:
        f.write(reasoning)