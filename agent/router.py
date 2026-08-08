from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
import uuid

from SQlDB.db import DatabaseConnection
from dbManagement import SQL_SERVER_CONNECTION_STRING, get_conversation_history, save_conversation, save_message
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

    # def classify(self, query: str,history:str) -> str:
    #     """
    #     تشخیص می‌دهد پرسش عمومی است یا فنی.
    #     خروجی فقط یکی از دو مقدار زیر است:

    #     technical
    #     general
    #     """
        
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": (
    #                 "Classify the user query as exactly one label.\n\n"

    #                 "technical: ATM, banking equipment, device errors, "
    #                 "troubleshooting, installation, configuration, maintenance, "
    #                 "printer, pinpad, dispenser, card reader, cash handling, "
    #                 "or operation.\n\n"

    #                 "general: greetings, small talk, unrelated questions, "
    #                 "or anything not technical.\n\n"

    #                 "Return exactly one word: technical or general."
    #             ),
    #         },
    #         {
    #             "role": "user",
    #             "content": query,
    #         },
    #     ]

    #     response = self.llm.chat(
    #         messages=messages,
    #         temperature=0,
    #     )

    #     result = (response.content or "").strip().lower()
    #     print("Resuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuut1111112323",flush=True)
    #     print(result,flush=True)
    #     if result not in {"technical", "general"}:
    #         return "technical"

    #     return result
    def classify(self, query: str, history: str | None = None) -> str:
        """
        تشخیص می‌دهد پرسش عمومی است یا فنی،
        با در نظر گرفتن history به‌صورت رشته.
        """

        system_prompt = (
            "Classify the user query as exactly one label.\n\n"

            "technical: ATM, banking equipment, device errors, "
            "troubleshooting, installation, configuration, maintenance, "
            "printer, pinpad, dispenser, card reader, cash handling, "
            "or operation.\n\n"

            "general: greetings, small talk, unrelated questions, "
            "or anything not technical.\n\n"

            "Use the conversation history only to understand the context "
            "of the current query.\n\n"

            "Return exactly one word: technical or general."
        )

        history_text = history.strip() if history else "No previous conversation."

        user_content = f"""
    Conversation history:
    --- HISTORY START ---
    {history_text}
    --- HISTORY END ---

    Current user query:
    --- QUERY START ---
    {query}
    --- QUERY END ---
    """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        response = self.llm.chat(
            messages=messages,
            temperature=0,
        )

        result = (response.content or "").strip().lower()

        print(
            "Classification result:",
            result,
            flush=True,
        )

        if result not in {"technical", "general"}:
            return "technical"

        return result

    def handle_stream(
        self,
        query: str,
        user_key:str,
        on_chunk: ChunkCallback,
        history: str,
        temperature: float = 0.1,
        
    ) -> None:
      
        intent = self.classify(query,history)
        print (intent,flush=True)
        if intent == "general":
            self.chat_agent.answer_stream(
                message=query,
                on_chunk=on_chunk,
                temperature=temperature,
                history=history
            )
            return

        self.document_agent.handle_stream(
            message=query,
           
            on_chunk=on_chunk,
            temperature=temperature,
            history=history,
        )
