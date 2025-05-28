# hybrid_search.py
import re
from pymorphy2 import MorphAnalyzer
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import tiktoken

load_dotenv()


class LLMReranker:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=10,
            max_retries=2
        )

    def rerank_documents(self, query: str, documents: List[Dict], llm_weight: float = 0.7) -> List[Dict]:
        results = []

        for doc in documents:
            # Оценка релевантности по смыслу
            semantic_score = self._get_semantic_similarity(query, doc["text"])

            # Оценка по ключевым словам (можно использовать существующий BM25 score)
            keyword_score = doc.get("bm25_score", 0.5)

            # Комбинированная оценка
            combined_score = (llm_weight * semantic_score +
                              (1 - llm_weight) * keyword_score)

            results.append({
                "document": doc,
                "score": combined_score
            })

        # Сортировка по убыванию общей оценки
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _truncate_text(self, text: str, max_tokens: int, model: str = "text-embedding-3-small") -> str:
        encoder = tiktoken.encoding_for_model(model)
        tokens = encoder.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return encoder.decode(tokens)
        return text

    def _get_semantic_similarity(self, text1: str, text2: str) -> float:
        # чтобы суммарно не превысить лимит (4096*2 = 8192)
        max_tokens_per_text = 4096
        text1 = self._truncate_text(text1, max_tokens_per_text)
        text2 = self._truncate_text(text2, max_tokens_per_text)
        emb = self.llm.embeddings.create(
            input=[text1, text2],
            model="text-embedding-3-small"
        )
        vec1 = np.array(emb.data[0].embedding)
        vec2 = np.array(emb.data[1].embedding)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class HybridSearch:
    def __init__(
        self,
        vector_store: Chroma,
        documents: List[Document],
        alpha: float = 0.5,
        top_k: int = 5,
        min_score: float = 0.3,
        K: int = 10,
        llm_rerank_samples: int = 15,
        llm_weight: float = 0.7
    ):
        self.alpha = alpha
        self.top_k = top_k
        self.min_score = min_score
        self.K = K
        self.llm_rerank_samples = llm_rerank_samples
        self.llm_weight = llm_weight

        self.morph = MorphAnalyzer()
        self.significant_pos = {"NOUN", "ADJF", "ADJS", "VERB"}

        self.documents = documents

        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            preprocess_func=self._preprocess_text,
            k=self.K
        )

        self.vector_store = vector_store
        self.reranker = LLMReranker()

    def _preprocess_text(self, text: str) -> List[str]:
        tokens = re.findall(r'\w+', text.lower())
        return [
            parsed.normal_form
            for token in tokens
            if (parsed := self.morph.parse(token)[0]).tag.POS in self.significant_pos
        ]

    @lru_cache(maxsize=128)
    def search(self, query: str) -> List[Document]:
        norm_query = " ".join(self._preprocess_text(query))

        with ThreadPoolExecutor() as executor:
            bm25_future = executor.submit(
                self.bm25_retriever.get_relevant_documents,
                norm_query
            )
            vector_future = executor.submit(
                self.vector_store.similarity_search_with_score,
                norm_query,
                self.K
            )

            bm25_docs = bm25_future.result()
            vector_results = vector_future.result()

        # Конвертация в словари для удобства
        combined = []
        for doc in bm25_docs:
            combined.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "bm25_score": 1.0,
                "vector_score": 0.0
            })

        for doc, score in vector_results:
            combined.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "bm25_score": 0.0,
                "vector_score": float(score)
            })

        # Первичная фильтрация
        filtered = [
            item for item in combined
            if (self.alpha * item["bm25_score"] +
                (1 - self.alpha) * item["vector_score"]) >= self.min_score
        ]

        # Выбираем топ документов для реранкинга
        top_for_rerank = sorted(
            filtered,
            key=lambda x: (self.alpha * x["bm25_score"] +
                           (1 - self.alpha) * x["vector_score"]),
            reverse=True
        )[:self.llm_rerank_samples]

        # LLM-реранкинг
        reranked = self.reranker.rerank_documents(
            query=query,
            documents=top_for_rerank,
            llm_weight=self.llm_weight
        )

        # Конвертация обратно в Document
        return [
            Document(
                page_content=item["document"]["text"],
                metadata=item["document"]["metadata"]
            )
            for item in reranked[:self.top_k]
        ]
