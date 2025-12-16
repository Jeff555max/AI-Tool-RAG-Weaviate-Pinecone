"""
Тест полного workflow RAG системы
"""

from embeddings.embedder import Embedder
from rag.retriever import Retriever
from rag.generator import RAGGenerator

print("=" * 80)
print("ТЕСТ RAG СИСТЕМЫ")
print("=" * 80)

# 1. Инициализация
print("\n[1/4] Инициализация компонентов...")
embedder = Embedder()
retriever = Retriever(embedder)
generator = RAGGenerator()
print("[OK] Компоненты инициализированы")

# 2. Загрузка документа
print("\n[2/4] Загрузка документа...")
with open("test_data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Разбиваем на параграфы
documents = [p.strip() for p in content.split("\n\n") if p.strip()]
print(f"[OK] Загружено {len(documents)} параграфов")

# 3. Добавление в векторную БД
print("\n[3/4] Добавление в Pinecone...")
retriever.add_documents(
    texts=documents,
    store_type="pinecone",
    metadata=[{"paragraph": i} for i in range(len(documents))]
)
print("[OK] Документы добавлены в векторную БД")

# 4. Тестовые вопросы
print("\n[4/4] Тестирование поиска и генерации ответов...")
print("=" * 80)

test_queries = [
    "Что такое Python?",
    "Расскажи про машинное обучение",
    "Что такое RAG?"
]

for query in test_queries:
    print(f"\n[Q] Вопрос: {query}")
    print("-" * 80)
    
    # Поиск релевантных документов
    results = retriever.retrieve(query, "pinecone", top_k=2)
    
    if results:
        # Генерация ответа
        answer = generator.generate_answer(query, results)
        
        print(f"\n[A] ОТВЕТ:\n{answer}\n")
        
        print(f"[S] ИСТОЧНИКИ ({len(results)} документов):")
        for i, doc in enumerate(results, 1):
            print(f"\n  [{i}] Релевантность: {doc['score']:.4f}")
            print(f"      {doc['text'][:150]}...")
    else:
        print("[FAIL] Документы не найдены")
    
    print("\n" + "=" * 80)

print("\n[SUCCESS] ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
