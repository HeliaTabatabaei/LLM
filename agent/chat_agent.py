class ChatAgent:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, message: str) -> str:
        system_prompt = "You are a helpful assistant."
        return self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=message,
            temperature=0.3
        )

    def answer_stream(self, message: str, on_chunk):
        system_prompt = "You are a helpful assistant."
        self.llm.chat_stream(
            system_prompt=system_prompt,
            user_prompt=message,
            temperature=0.3,
            on_chunk=on_chunk
        )
