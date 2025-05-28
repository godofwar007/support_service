# db.py
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

PERSIST_DIRECTORY = "./vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400

# Предположим, что у нас есть папка "knowlage", где будут лежать все документы.
# Если нужно другое название, измените KNOWLAGE_DIR здесь:
KNOWLAGE_DIR = os.path.join(os.path.dirname(__file__), "knowlage")


# Загружает все файлы из директории (txt, doc, docx)
# и возвращает список документов (до разбиения на chunks).
def load_files_from_directory(directory_path: str):
    all_docs = []
    if not os.path.exists(directory_path):
        print(f"Директория '{directory_path}' не найдена!")
        return all_docs

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Пропускаем папки
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)

        except Exception as e:
            print(f"Ошибка при загрузке файла {filename}: {e}")

    return all_docs


# ЗАГРУЗКА/ИНДЕКСАЦИЯ ДОКУМЕНТОВ
def create_docs_for_indexing():

    # Загружаем документы из директории
    all_docs = load_files_from_directory(KNOWLAGE_DIR)

    # Разбиваем документы на фрагменты (chunks)
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(all_docs)

    processed_docs = []
    for d in docs:
        new_doc = Document(
            page_content=d.page_content,
            metadata={"original_text": d.page_content}
        )
        processed_docs.append(new_doc)

    return docs, processed_docs


def create_chroma(docs):
    print("Создаём новое Chroma-хранилище...")
    embedding = OpenAIEmbeddings(openai_api_key=API_KEY)
    index = Chroma.from_documents(
        docs,
        embedding,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"Индекс создан: {len(docs)} фрагментов сохранено.")
    return index


def create_or_load_chroma(docs):

    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    try:
        index = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings(openai_api_key=API_KEY)
        )
        doc_count = index._collection.count()
        if doc_count == 0:
            raise ValueError("В хранилище нет документов. Пересоздаём индекс.")
        print(
            f"Загружено существующее хранилище Chroma. "
            f"Документов: {doc_count}"
        )
        return index
    except Exception:
        print("Ошибка загрузки Chroma. Создаём новое хранилище...")
        return create_chroma(docs)


# Динамическое обновление векторного хранилища,

def refresh_chroma_index():

    print("Обновляем (пересоздаём) векторное хранилище на основе текущих файлов...")
    docs, processed_docs = create_docs_for_indexing()

    new_index = create_chroma(docs)

    return new_index
