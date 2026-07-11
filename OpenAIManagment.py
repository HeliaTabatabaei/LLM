from typing import List

from openai import OpenAI

from config import EMBED_MODEL, LLM_MODEL, OPENAI_API_KEY

from typing import List, Optional

from qdrant_client import models

from Models.mainModels import SearchFilters
from prompts_config import SYSTEM_PROMPT, USER_PROMPT
# Initialize clients
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.gapgpt.app/v1"
)

def embed_query(text: str) -> List[float]:
    """تبدیل متن به embedding vector"""
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding
def createSummery(old_summary,new_user_msg,new_assistant_msg):
    summary_prompt = f"""
            با توجه به خلاصه قبلی مکالمه و پیام‌های جدید مبادله شده، یک خلاصه کوتاه، جامع و به زبان فارسی از کل مکالمه تا این لحظه بنویس. 
            جزئیات فنی مهم (مانند نام ابزارها، پورت‌ها، خطاها یا تصمیمات کلیدی) را حفظ کن اما خلاصه را تا حد امکان فشرده نگه‌دار.

            خلاصه قبلی:
            {old_summary}

            پیام‌های جدید:
            کاربر: {new_user_msg}
            دستیار: {new_assistant_msg}

            خلاصه جدید به‌روزشده:
            """

            # ۳. فراخوانی مدل برای خلاصه‌سازی (یک مدل سبک‌تر و سریع‌تر ترجیح داده می‌شود)
    response = client.responses.create(
                model=LLM_MODEL,  # یا یک مدل سریع‌تر/ارزان‌تر
                input=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            )
    return response.output_text.strip()
def CreateResponse(context,query,history,temperature):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                context=context,
                query=query,
                history=history
            )
        }
    ]
    response = client.responses.create(
        model=LLM_MODEL,
        input=messages,
        temperature=temperature
    )

    answer = response.output_text
    return response
def CreateResponseWithInpute(messages,temperature):

    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {
    #         "role": "user",
    #         "content": USER_PROMPT.format(
    #             context=context,
    #             query=query,
    #             history=history
    #         )
    #     }
    # ]
    response = client.responses.create(
        model=LLM_MODEL,
        input=messages,
        temperature=temperature
    )

    answer = response.output_text
    return response
