import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["TOKENHUB_API_KEY"],
    base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
)


messages = [{"role": "system", "content": "Always reply in English. Do not generate images."}]

def ask(user_text, verbose=False):
    print('> {0}'.format(user_text))
    messages.append({"role": "user", "content": user_text})
    stream = client.chat.completions.create(
        model="hy4", messages=messages, max_tokens=2000, stream=True
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            t = chunk.choices[0].delta.content
            print(t, end="", flush=True)
            parts.append(t)
    print()
    messages.append({"role": "assistant", "content": "".join(parts)})

ask("hello")
ask("What's the day today?")
ask("What did I just say?")  # 履歴があるので答えられる