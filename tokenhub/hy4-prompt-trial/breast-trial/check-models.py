import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


c = OpenAI(api_key=os.environ["TOKENHUB_API_KEY"],
           base_url="https://tokenhub-intl.tencentcloudmaas.com/v1")
for m in c.models.list():
    print(m.id)