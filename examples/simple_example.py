"""
Простой пример использования RAG системы.
Этот скрипт показывает базовый workflow: добавление документов и поиск.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embedder import Embedder
from rag.retriever import Retriever


def main():
    print("\n" + "="*60)
    print("  Простой пример использования RAG")
    print("="*60 + "\n")
    
    # Шаг 1: Инициализация
    print("[1/4] Инициализация компонентов...")
    embedder = Embedder()
    retriever = Retriever(embedder)
    
    # Шаг 2: Подготовка документов
    print("[2/4] Подготовка документов...")
    documents = [
        "Python - это высокоуровневый язык программирования общего назначения",
        "JavaScript используется для создания интерактивных веб-страниц",
        "SQL - язык структурированных запросов для работы с базами данных",
        "HTML - язык разметки для создания веб-страниц",
        "CSS используется для стилизации HTML элементов"
    ]
    
    print(f"   Подготовлено {len(documents)} документов")
    
    # Шаг 3: Добавление в векторную БД
    print("[3/4] Добавление документов в Pinecone...")
    retriever.add_documents(
        texts=documents,
        store_type="pinecone",
        metadata=[{"doc_id": i, "category": "programming"} for i in range(len(documents))]
    )
    print("   Документы успешно добавлены!")
    
    # Шаг 4: Поиск
    print("[4/4] Выполнение поисковых запросов...\n")
    
    queries = [
        "Какой язык используется для веб-разработки?",
        "Что такое Python?",
        "Как работать с базами данных?"
    ]
    
    for query in queries:
        print(f"\nВопрос: {query}")
        print("-" * 60)
        
        results = retriever.retrieve(
            query=query,
            store_type="pinecone",
            top_k=2
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n  Результат {i} [Релевантность: {result['score']:.4f}]")
            print(f"  {result['text']}")
    
    # Cleanup
    print("\n" + "="*60)
    print("  Завершение работы")
    print("="*60)
    retriever.cleanup()
    
    print("\n[SUCCESS] Пример выполнен успешно!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Прервано пользователем")
    except Exception as e:
        print(f"\n[ERROR] Ошибка: {e}")
