import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["TOKENHUB_API_KEY"],
    base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
)

stream = client.chat.completions.create(
    model="hy4",
    messages=[
        {"role": "system", "content": "Always reply in English. Do not generate images."},
        {"role": "user", "content": "hello"},
    ],
    max_tokens=200,
    stream=True,
)

parts = []
finish_reason = None

for chunk in stream:  # ここはストリームが終わるまでブロックされる
    if not chunk.choices:
        continue
    choice = chunk.choices[0]
    if choice.delta.content:
        print(choice.delta.content, end="", flush=True)
        parts.append(choice.delta.content)
    if choice.finish_reason:
        finish_reason = choice.finish_reason

print()

# ここに来た時点で生成は完了している
full_text = "".join(parts)
print(f"[finish_reason={finish_reason}, chars={len(full_text)}]")

with open("hy4_output.txt", "w", encoding="utf-8") as f:
    f.write(full_text)