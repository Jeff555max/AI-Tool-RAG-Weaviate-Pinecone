"""
Скрипт для очистки индексов в векторных БД
"""

from stores.pinecone_store import PineconeStore
from stores.weaviate_store import WeaviateStore
from embeddings.embedder import Embedder

print("Очистка векторных БД...")

# Pinecone
try:
    embedder = Embedder()
    pinecone_store = PineconeStore(embedder=embedder)
    pinecone_store.delete_index()
    print("[OK] Pinecone индекс удален")
except Exception as e:
    print(f"[FAIL] Ошибка при удалении Pinecone: {e}")

# Weaviate
try:
    weaviate_store = WeaviateStore(embedder=embedder)
    # Удаляем коллекцию
    if weaviate_store.client.collections.exists("RAGDocument"):
        weaviate_store.client.collections.delete("RAGDocument")
        print("[OK] Weaviate коллекция удалена")
    weaviate_store.close()
except Exception as e:
    print(f"[FAIL] Ошибка при удалении Weaviate: {e}")

print("\nГотово! Теперь можете добавить документы заново через GUI.")
