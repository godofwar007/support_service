# history.py
import os
import json
from langchain.schema import HumanMessage, AIMessage

CHAT_HISTORIES_FILE = os.path.join(
    os.path.dirname(__file__), "chat_histories.json"
)


def load_chat_histories():
    if os.path.exists(CHAT_HISTORIES_FILE):
        with open(CHAT_HISTORIES_FILE, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if content:
                data = json.loads(content)
                return {int(k): v for k, v in data.items()}
    return {}


def save_chat_histories(chat_histories):
    data = {str(k): v for k, v in chat_histories.items()}
    with open(CHAT_HISTORIES_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def format_chat_history_as_messages(chat_log):

    messages = []
    for question, answer in chat_log:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages
