import os
import anthropic

from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
print(client.files.list())

