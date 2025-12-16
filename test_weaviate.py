"""
Диагностика Weaviate
"""
from embeddings.embedder import Embedder
from stores.weaviate_store import WeaviateStore

print("=== ТЕСТ WEAVIATE ===\n")

# 1. Подключение
print("[1] Подключение к Weaviate...")
embedder = Embedder()
store = WeaviateStore(embedder=embedder)
print("[OK] Подключено\n")

# 2. Добавление тестовых документов
print("[2] Добавление документов...")
test_docs = [
    "Python - это язык программирования",
    "Machine Learning - это раздел искусственного интеллекта",
    "RAG означает Retrieval-Augmented Generation"
]

store.add_texts(test_docs)
print("[OK] Документы добавлены\n")

# 3. Проверка количества
print("[3] Проверка количества документов...")
collection = store.client.collections.get(store.class_name)
response = collection.aggregate.over_all(total_count=True)
print(f"[OK] Всего документов: {response.total_count}\n")

# 4. Тестовый поиск
print("[4] Тестовый поиск...")
query = "Что такое Python?"
results = store.query(query, top_k=3)

print(f"Запрос: {query}")
print(f"Найдено: {len(results)} результатов\n")

for i, r in enumerate(results, 1):
    print(f"[{i}] Score: {r['score']:.4f}")
    print(f"    Text: {r['text']}")
    print()

store.close()
print("=== ТЕСТ ЗАВЕРШЕН ===")
