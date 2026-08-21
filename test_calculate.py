import os
import time

from providers import create_provider
from providers.openai_provider import OpenAIProvider

start= time.time()
print(start)


provider = create_provider(
provider_name="openai",
base_uri="https://api.gapgpt.app/v1",
api_key="sk-bVyAADjnBq7laj5jYKkhxFCA2W6iGhBC7dNUfdka0b99wiRw",
model="gpt-4.1-mini",
embed_model="text-embedding-3-large",
)

provider.embed_query("سلام خوبی من جواد سرلک هستم مهندس نرم افزار هستم چی میگی ب خودت؟ خودت ؟ بببی بلبیبیبیب لبلبل ل")
print(f"Take:: ",time.time()-start)