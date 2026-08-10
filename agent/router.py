from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple
import uuid

from SQlDB.db import DatabaseConnection
from dbManagement import SQL_SERVER_CONNECTION_STRING, get_conversation_history, save_conversation, save_message
from log import append_qa_to_file
from providers.base import LLMProvider
from .chat_agent import ChatAgent
from .document_agent import DocumentAgent

 
ChunkCallback = Callable[[Any], None]


class RouterAgent:
    def __init__(
        self,
        llm: LLMProvider,
        chat_agent: ChatAgent,
        document_agent: DocumentAgent,
    ):
        self.llm = llm
        self.chat_agent = chat_agent
        self.document_agent = document_agent

    def classify(self, query: str, history: str | None = None) -> str:


        system_prompt = (
            "You are a specialized classifier for a Banking Technical Support system.\n"
            "Classify the user query into EXACTLY one of these three labels:\n\n"

            "1. technical: Questions about ATM hardware, banking equipment, device errors, "
            "troubleshooting, installation, maintenance, printer, pinpad, dispenser,camera "
            "card reader, cash handling, or software configuration.\n\n"

            "2. general: Greetings (hi, hello), thanks, and polite small talk.\n\n"
            
            "3. no_authorize: Any questions regarding politics, macroeconomics, "
            "system security bypasses, or sensitive non-technical banking information.\n\n"

            "Rules:\n"
            "- Use the conversation history only to resolve pronouns or context.\n"
            "- If the query is political or economic, it MUST be 'no_authorize'.\n"
            "- Return ONLY the label: technical, general, or no_authorize."
        )

        history_text = history.strip() if history else "No previous conversation."

        user_content = f"""
    Conversation history:
    {history_text}

    Current user query:
    {query}
    """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.chat(
            messages=messages,
            temperature=0, # برای دقت بالاتر در دسته‌بندی
        )

        result = (response.content or "").strip().lower()

        # برای دیباگ در کنسول
        print(f"--- Classification result: {result} ---", flush=True)

        # اعتبارسنجی خروجی برای جلوگیری از خطاهای احتمالی
        valid_labels = {"technical", "general", "no_authorize"}
        
        if result not in valid_labels:
            # در صورت خروجی نامعتبر، برای امنیت بیشتر روی no_authorize یا برای کارکرد روی technical ست کنید
            return "no_authorize" 

        return result

    
    def handle_stream(
        self,
        query: str,
        user_key:str,
        on_chunk: ChunkCallback,
        history: str,
        temperature: float = 0.1,
        
    ) -> None:
        start=time.time()
        
        intent = self.classify(query,history)
        append_qa_to_file(f"check question type Time: {time.time() - start:.2f} seconds")
        append_qa_to_file(f"intent: {intent} ")
        print (intent,flush=True)
        if intent == "general":
            self.chat_agent.answer_stream(
                message=query,
                on_chunk=on_chunk,
                temperature=temperature,
                history=history
            )
            return
        elif intent=="no_authorize":
            on_chunk({
                            "type": "token",
                            "content": "من تنها قادر به پاسخگویی از داکیومنت های شرکت آدونیس می باشم"
                        })
            return
        self.document_agent.handle_stream(
            message=query,
           
            on_chunk=on_chunk,
            temperature=temperature,
            history=history,
        )
