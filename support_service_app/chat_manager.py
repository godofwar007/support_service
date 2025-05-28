# chat_manager.py
from typing import List
from langchain.schema import Document
from .ai_settings import llm, prompt_template
from .history import load_chat_histories, save_chat_histories


class ChatManager:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def _needs_clarification(self, query: str) -> bool:
        return len(query.split()) < 3

    def _get_clarification(self, query: str) -> str:
        return f"Уточните ваш запрос: '{query}'. Какие аспекты вас интересуют?"

    def _format_context(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"[Источник: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def process_query(self, chat_id: int, query: str) -> str:
        chat_logs = load_chat_histories()
        if chat_id not in chat_logs:
            chat_logs[chat_id] = []
        chat_history = chat_logs[chat_id]

        if self._needs_clarification(query):
            response = self._get_clarification(query)
            chat_history.append((query, response))
            save_chat_histories(chat_logs)
            return response

        docs = self.search_engine.search(query)
        context = self._format_context(docs)

        # Берем последние 3 сообщения из истории
        history = chat_history[-3:]
        history_str = "\n".join(f"User: {q}\nBot: {a}" for q, a in history)

        prompt = prompt_template.format(
            chat_history=history_str,
            context=context,
            question=query
        )

        response = llm.invoke(prompt).content
        chat_history.append((query, response))
        save_chat_histories(chat_logs)

        return response
