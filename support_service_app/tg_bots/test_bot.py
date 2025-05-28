# test_bot.py
import telebot
import re
import os
import logging
from dotenv import load_dotenv
from support_service_app.chat_manager import ChatManager
from support_service_app.hybrid_search import HybridSearch
from support_service_app.history import load_chat_histories
from support_service_app.history import save_chat_histories
from support_service_app.db import create_docs_for_indexing
from support_service_app.db import create_or_load_chroma
# Настройка логирования
logging.basicConfig(level=logging.INFO)

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TEST_BOT_API")

# Инициализация документов и движка поиска
original_docs, _ = create_docs_for_indexing()
my_vector_store = create_or_load_chroma(
    original_docs)  # Создаем или загружаем хранилище
search_engine = HybridSearch(
    vector_store=my_vector_store,
    documents=original_docs,
    K=10,  # Ищет по 10 документов (вектор+БМ25)
    alpha=0.5,  # общая оценка: 0.5*BM25_score + 0.5*vector_score
    min_score=0.4,  # Отбрасывает документы с оценкой < 0.4
    llm_rerank_samples=10,  # сколько документов отправить на проверку
    llm_weight=0.6,          # Сила доверия к ИИ. 0.7 = 70%
    top_k=5  # Возвращает топ-5 документов
)

chat_manager = ChatManager(search_engine)

bot = telebot.TeleBot(BOT_TOKEN)
chat_histories = load_chat_histories()
phone_pattern = re.compile(r"(\+?\d[\d\-\s()]{5,}\d)")
MAX_HISTORY = 20


@bot.message_handler(commands=['reset', 'start'])
def reset_history(message):
    chat_id = message.chat.id
    chat_histories[chat_id] = []
    save_chat_histories(chat_histories)
    bot.send_message(chat_id, "История сброшена. Чем могу помочь?")


@bot.message_handler(content_types=['text'])
def main(message):
    chat_id = message.chat.id
    user_text = message.text.strip()
    has_phone = bool(phone_pattern.search(user_text))

    # Загружаем историю из файла
    chat_histories = load_chat_histories()
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
        save_chat_histories(chat_histories)

    if has_phone:
        reply = "Спасибо! Мы свяжемся с вами в ближайшее время."
    else:
        # process_query уже обновляет историю в файле
        reply = chat_manager.process_query(chat_id, user_text)

    # Ограничиваем историю по количеству сообщений
    chat_histories = load_chat_histories()  # обновляем данные после process_query
    if len(chat_histories.get(chat_id, [])) > MAX_HISTORY:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]
        save_chat_histories(chat_histories)

    bot.send_message(chat_id, reply)


bot.polling(non_stop=True)
