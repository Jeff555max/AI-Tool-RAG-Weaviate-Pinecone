"""
Проверка содержимого векторной БД
"""

from embeddings.embedder import Embedder
from rag.retriever import Retriever

print("Проверка содержимого векторной БД...")
print("=" * 80)

embedder = Embedder()
retriever = Retriever(embedder)

# Тестовый запрос
query = "Python"
print(f"\nЗапрос: {query}\n")

# Получаем результаты из Pinecone
results = retriever.retrieve(query, "pinecone", top_k=3)

print(f"Найдено документов: {len(results)}\n")

for i, result in enumerate(results, 1):
    print(f"Документ {i}:")
    print(f"Релевантность: {result['score']:.4f}")
    print(f"Текст: {result['text'][:200]}")
    print("-" * 80)
    print()

print("\nЕсли видите английский текст - запустите clear_index.py")
print("Затем добавьте русские документы заново через GUI")
