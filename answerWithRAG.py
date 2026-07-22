
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import BackgroundTasks

from OpenAIManagment import CreateResponse, CreateResponseStream, CreateResponseWithInput, embed_query
from QdrantManagment import build_context, save_message_to_qdrant, search_chat_history
from db import DatabaseConnection
from dbManagement import SQL_SERVER_CONNECTION_STRING, get_conversation_history, save_assistant_message_task, save_conversation, save_message, update_conversation_summary_task
from prompts_config import SYSTEM_PROMPT, USER_PROMPT
from fastapi import BackgroundTasks
def normalize_conversation_id(conversation_id: Optional[str]) -> Tuple[str, bool]:
    """
    اگر conversation_id معتبر باشد:
        (conversation_id, False)
    اگر خالی/نامعتبر باشد:
        (new_uuid, True)
    """
    try:
        if conversation_id is None:
            raise ValueError

        conversation_id = str(conversation_id).strip()

        if conversation_id in ("", "undefined", "null", "None"):
            raise ValueError

        normalized = str(uuid.UUID(conversation_id))
        return normalized, False

    except (ValueError, TypeError, AttributeError):
        return str(uuid.uuid4()), True




def answer_with_rag_withHistoryAndVectorDB(
    query: str,
    
    results: List[Any],
    background_tasks: BackgroundTasks,
    temperature: float = 0.1,
    conversation_id: Optional[str] = None,
    user_key: Optional[str] = None
) -> Dict[str, Any]:
    conversation_id, is_new_chat = normalize_conversation_id(conversation_id)
  
    start = time.time()
    context = build_context(results)
    print(f"1 build_context ( {time.time() - start:.2f} seconds")
    
    # 1) تولید embedding برای query جدید جهت جستجوی memory در Qdrant
    try:
        query_vector = embed_query(query)
    except Exception as e:
        #log_message(f"embed_query failed in history retrieval: {e}")
        query_vector = None
    print(f"2 build_context ( {time.time() - start:.2f} seconds")
    # 2) جستجوی حافظه بلندمدت از Qdrant
    relevant_history_text = ""
    
    if query_vector is not None:
        try:
            relevant_history_text = search_chat_history(
                query_vector=query_vector,
                conversation_id=conversation_id,
                limit=3
            )
        except Exception as e:
            #log_message(f"search_chat_history failed: {e}")
            relevant_history_text = ""
    #log_message("step4")
    print(f"3 search_chat_history ( {time.time() - start:.2f} seconds")
    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        if not is_new_chat:
            cursor.execute(
                "SELECT 1 FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            if not cursor.fetchone():
                is_new_chat = True

        if is_new_chat:
            conversation_id = save_conversation(
                cursor=cursor,
                conversation_id=conversation_id,
                title=query,
                user_key=user_key,
                model_id=1
            )

        # 3) گرفتن 6 پیام آخر از SQL Server
        history = get_conversation_history(
            cursor=cursor,
            conversation_id=conversation_id,
            limit=6
        )

        # 4) ذخیره پیام کاربر در SQL Server
        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="user",
            content=query
        )
        print(f"4 save sql ( {time.time() - start:.2f} seconds")
   # log_message("step saveeeeeeeeeeeeeeeeee")
    # 5) ذخیره پیام کاربر در Qdrant برای استفاده‌های بعدی
    try:
        background_tasks.add_task(
        save_message_to_qdrant,
            conversation_id=conversation_id,
            role="user",
            content=query,
            user_key=user_key
        )
    except Exception as e:
        print(f"error  save_message_to_qdrant:{e}")
        #log_message(f"save_message_to_qdrant(user) failed: {e}")

    # 6) ساخت متن history کوتاه‌مدت
    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]
    )

    # 7) ساخت متن حافظه بلندمدت
    long_term_memory_text = (
        relevant_history_text.strip()
        if relevant_history_text.strip()
        else "یادآوری مرتبطی از حافظه بلندمدت پیدا نشد."
    )

    combined_memory = f"""
    حافظه بلندمدت مرتبط از Qdrant:
        {long_term_memory_text}

    تاریخچه اخیر گفتگو:
        {history_text}
    """.strip()

    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {
    #         "role": "user",
    #         "content": USER_PROMPT.format(
    #             context=context,
    #             query=query,
    #             history=combined_memory
    #         )
    #     }
    # ]
    # response = client.responses.create(
    #     model=LLM_MODEL,
    #     input=messages,
    #     temperature=temperature
    # )

    # answer = response.output_text
    response=CreateResponse(context,query,combined_memory,temperature)
    answer = response.output_text
    response_id = getattr(response, "id", None)

    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            provider_response_id=response_id
        )

    # 9) ذخیره پاسخ دستیار در Qdrant
    try:
        background_tasks.add_task(
            save_message_to_qdrant,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            user_key=user_key
        )
    except Exception as e:
        print(f"save_message_to_qdrant(assistant) failed: {e}")
        #log_message(f"save_message_to_qdrant(assistant) failed: {e}")

    print(f"4  end answer with rag ( {time.time() - start:.2f} seconds")
    
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "provider_response_id": response_id
    }

def answer_with_rag_withHistory(
    query: str,
    results: List[Any],
    temperature: float = 0.1,
    conversation_id: Optional[str] = None,
    user_key: Optional[str] = None
) -> Dict[str, Any]:
    conversation_id, is_new_chat = normalize_conversation_id(conversation_id)
    
    context = build_context(results)

    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        if not is_new_chat:
            cursor.execute(
                "SELECT 1 FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            if not cursor.fetchone():
                is_new_chat = True


        if is_new_chat:
            conversation_id=save_conversation(
                cursor=cursor,
                conversation_id=conversation_id,
                title=query,
                user_key=user_key,
                model_id=1
            )

        history = get_conversation_history(
            cursor=cursor,
            conversation_id=conversation_id,
            limit=6
        )

        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="user",
            content=query
        )
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    response=CreateResponse(context,query,history_text,temperature)
    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     *history,
    #     {
    #         "role": "user",
    #         "content": USER_PROMPT.format(
    #             context=context,
    #             query=query,
    #             history=history_text
                
    #         )
    #     }
    # ]
    # #log_message(messages)
    # response = client.responses.create(
    #     model=LLM_MODEL,
    #     input=messages,
    #     temperature=temperature
    # )

    answer = response.output_text
    response_id = getattr(response, "id", None)

    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            provider_response_id=response_id
        )
 
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "provider_response_id": response_id
    }

def answer_with_rag(query: str, results, temperature: float = 0.1) -> str:
    """تولید پاسخ با RAG"""
    context = build_context(results)
    
    # system_prompt = SYSTEM_PROMPT
    # user_prompt = USER_PROMPT.format(
    #     query=query,
    #     context=context,
    #     history=""
    # )
    response=CreateResponse(context,query,"-",temperature)
    # response = client.responses.create(
    #     model=LLM_MODEL,
    #     temperature=temperature,
    #     input=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    # )
    #print(response.output_text)

# Get token usage
    # print(response.usage)
    # print("Input tokens:", response.usage.input_tokens)
    # print("Output tokens:", response.usage.output_tokens)
    # print("Total tokens:", response.usage.total_tokens)
    # print("id:", response.id)
    return response.output_text
def answer_with_rag_stream(
    query: str,
    results,
    temperature: float = 0.1
):
    """
    تولید پاسخ Stream با RAG
    """
    context = build_context(results)

    yield from CreateResponseStream(
        context=context,
        query=query,
        history="-",
        temperature=temperature
    )
def answer_with_rag_with_summary(
        query: str,
        results: List[Any],
        background_tasks: BackgroundTasks,
        temperature: float = 0.1,
        conversation_id: Optional[str] = None,
        user_key: Optional[str] = None
) -> Dict[str, Any]:
    conversation_id, is_new_chat = normalize_conversation_id(conversation_id)
    context = build_context(results)
    current_summary = ""
    history = []

    start_db = time.time()
    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        if not is_new_chat:
            cursor.execute(
                "SELECT 1 FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            if not cursor.fetchone():
                is_new_chat = True

        if is_new_chat:
            save_conversation(
                cursor=cursor,
                conversation_id=conversation_id,
                title=query,
                user_key=user_key,
                model_id=1
            )
        else:
            cursor.execute(
                "SELECT Summary FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                current_summary = row[0]

        history = get_conversation_history(
            cursor=cursor,
            conversation_id=conversation_id,
            limit=4
        )

        save_message(cursor, conversation_id, "user", query)
    print(f"DB First Operations Time: {time.time() - start_db:.2f} seconds")

    prompt_content = USER_PROMPT.format(
        context=context,
        history=current_summary if current_summary else "No previous history.",
        query=query
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": prompt_content}
    ]

    start_llm = time.time()
    # response = client.responses.create(
    #     model=LLM_MODEL,
    #     input=messages,
    #     temperature=temperature
    # )

    response=CreateResponseWithInput(messages,temperature)
    print(f"LLM Generation Time: {time.time() - start_llm:.2f} seconds")

    answer = response.output_text
    response_id = getattr(response, "id", None)

    background_tasks.add_task(save_assistant_message_task, conversation_id, answer, response_id)

    background_tasks.add_task(
        update_conversation_summary_task,
        conversation_id=conversation_id,
        new_user_msg=query,
        new_assistant_msg=answer
    )

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "provider_response_id": response_id
    }
def answer_stream_only_llm(
    query: str,
    temperature: float = 0.1
):
    """
    فقط از LLM پاسخ می‌گیرد، بدون RAG
    """
    yield from CreateResponseStream(
        context="",
        query=query,
        history="-",
        temperature=temperature
    )