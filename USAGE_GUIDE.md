# Руководство по использованию AI-Tool-RAG-Weaviate-Pinecone

## Содержание
1. [Быстрый старт](#быстрый-старт)
2. [Основные сценарии использования](#основные-сценарии-использования)
3. [Работа с документами](#работа-с-документами)
4. [Семантический поиск](#семантический-поиск)
5. [Продвинутые примеры](#продвинутые-примеры)

---

## Быстрый старт

### 1. Запуск демо
```bash
# Активируйте виртуальное окружение
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Запустите демо
python examples/demo_usage.py
```

Демо покажет:
- Добавление документов в векторные БД
- Выполнение поисковых запросов
- Сравнение результатов между Pinecone и Weaviate
- Работу с чанкингом текста

---

## Основные сценарии использования

### Сценарий 1: Создание базы знаний

Создайте файл `my_knowledge_base.py`:

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

# Инициализация
embedder = Embedder()
retriever = Retriever(embedder)

# Ваши документы (статьи, FAQ, документация)
documents = [
    "Python - высокоуровневый язык программирования",
    "Machine Learning - раздел искусственного интеллекта",
    "RAG комбинирует поиск информации с генерацией текста"
]

# Добавляем в Pinecone
retriever.add_documents(
    texts=documents,
    store_type="pinecone",
    metadata=[
        {"source": "wiki", "topic": "programming"},
        {"source": "wiki", "topic": "ai"},
        {"source": "docs", "topic": "rag"}
    ]
)

print("База знаний создана!")
```

Запуск:
```bash
python my_knowledge_base.py
```

---

### Сценарий 2: Поиск по базе знаний

Создайте файл `search_knowledge.py`:

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

embedder = Embedder()
retriever = Retriever(embedder)

# Ваш вопрос
query = "Что такое машинное обучение?"

# Поиск в Pinecone
results = retriever.retrieve(
    query=query,
    store_type="pinecone",
    top_k=3  # Топ-3 результата
)

# Вывод результатов
print(f"\nВопрос: {query}\n")
for i, result in enumerate(results, 1):
    print(f"{i}. [Score: {result['score']:.4f}]")
    print(f"   {result['text'][:100]}...\n")
```

Запуск:
```bash
python search_knowledge.py
```

---

### Сценарий 3: Сравнение векторных БД

Создайте файл `compare_stores.py`:

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

embedder = Embedder()
retriever = Retriever(embedder)

# Сравнить результаты поиска в разных БД
query = "Explain neural networks"

retriever.compare_stores(
    query=query,
    top_k=3
)

# Это покажет результаты из Pinecone и Weaviate рядом
```

---

## Работа с документами

### Добавление документов из файла

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

embedder = Embedder()
retriever = Retriever(embedder)

# Читаем документы из файла
with open("my_documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Очищаем от пустых строк
documents = [doc.strip() for doc in documents if doc.strip()]

# Добавляем в векторную БД
retriever.add_documents(
    texts=documents,
    store_type="weaviate"
)

print(f"Добавлено {len(documents)} документов")
```

### Работа с большими текстами (чанкинг)

```python
from utils.chunker import TextChunker
from embeddings.embedder import Embedder
from rag.retriever import Retriever

# Инициализация
chunker = TextChunker(chunk_size=512, chunk_overlap=50)
embedder = Embedder()
retriever = Retriever(embedder)

# Большой текст
with open("long_article.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

# Разбиваем на чанки
chunks = chunker.chunk_text(long_text)
print(f"Текст разбит на {len(chunks)} частей")

# Добавляем чанки в БД
retriever.add_documents(
    texts=chunks,
    store_type="pinecone",
    metadata=[{"chunk_id": i, "source": "article"} for i in range(len(chunks))]
)
```

---

## Семантический поиск

### Простой поиск

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

embedder = Embedder()
retriever = Retriever(embedder)

# Поиск
results = retriever.retrieve(
    query="Как работает нейронная сеть?",
    store_type="pinecone",
    top_k=5
)

# Обработка результатов
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result.get('metadata', {})}")
    print("-" * 50)
```

### Поиск с фильтрацией (Pinecone)

```python
# Поиск только в определенной категории
results = retriever.stores["pinecone"].query(
    query_text="machine learning",
    top_k=5,
    filter_dict={"topic": "ai"}  # Только документы с topic=ai
)
```

### Многоязычный поиск

```python
# OpenAI embeddings поддерживают множество языков
queries = [
    "What is Python?",           # Английский
    "Что такое Python?",         # Русский
    "Was ist Python?",           # Немецкий
]

for query in queries:
    results = retriever.retrieve(query, "pinecone", top_k=1)
    print(f"{query} -> {results[0]['text'][:50]}...")
```

---

## Продвинутые примеры

### Пример 1: RAG-система для FAQ

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

class FAQSystem:
    def __init__(self):
        self.embedder = Embedder()
        self.retriever = Retriever(self.embedder)
    
    def load_faq(self, faq_file):
        """Загрузить FAQ из файла"""
        with open(faq_file, "r", encoding="utf-8") as f:
            faqs = [line.strip() for line in f if line.strip()]
        
        self.retriever.add_documents(
            texts=faqs,
            store_type="pinecone",
            metadata=[{"type": "faq"} for _ in faqs]
        )
        print(f"Загружено {len(faqs)} вопросов")
    
    def answer_question(self, question):
        """Найти ответ на вопрос"""
        results = self.retriever.retrieve(
            query=question,
            store_type="pinecone",
            top_k=1
        )
        
        if results:
            return results[0]['text']
        return "Ответ не найден"

# Использование
faq = FAQSystem()
faq.load_faq("faq.txt")
answer = faq.answer_question("Как установить Python?")
print(answer)
```

### Пример 2: Поиск похожих документов

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

embedder = Embedder()
retriever = Retriever(embedder)

def find_similar_documents(document_text, top_k=5):
    """Найти документы похожие на данный"""
    results = retriever.retrieve(
        query=document_text,
        store_type="weaviate",
        top_k=top_k
    )
    
    return [
        {
            "text": r['text'],
            "similarity": r['score']
        }
        for r in results
    ]

# Использование
my_doc = "Статья о глубоком обучении и нейронных сетях"
similar = find_similar_documents(my_doc, top_k=3)

for i, doc in enumerate(similar, 1):
    print(f"{i}. Similarity: {doc['similarity']:.4f}")
    print(f"   {doc['text'][:80]}...\n")
```

### Пример 3: Batch обработка запросов

```python
from embeddings.embedder import Embedder
from rag.retriever import Retriever

embedder = Embedder()
retriever = Retriever(embedder)

# Множество запросов
queries = [
    "What is AI?",
    "Explain machine learning",
    "How do neural networks work?",
    "What is deep learning?"
]

# Обработка всех запросов
all_results = {}
for query in queries:
    results = retriever.retrieve(query, "pinecone", top_k=2)
    all_results[query] = results

# Вывод результатов
for query, results in all_results.items():
    print(f"\nQuery: {query}")
    print(f"Best match: {results[0]['text'][:60]}...")
    print(f"Score: {results[0]['score']:.4f}")
```

### Пример 4: Создание собственного эмбеддера

```python
from embeddings.embedder import Embedder

# Использование разных моделей OpenAI
embedder_small = Embedder(model="text-embedding-3-small")  # Быстрее, дешевле
embedder_large = Embedder(model="text-embedding-3-large")  # Точнее, дороже

# Сравнение
text = "Machine learning example"
emb_small = embedder_small.embed_text(text)
emb_large = embedder_large.embed_text(text)

print(f"Small model dimension: {len(emb_small)}")
print(f"Large model dimension: {len(emb_large)}")
```

---

## Полезные команды

### Проверка настройки
```bash
python scripts/check_setup.py
```

### Очистка индексов
```python
from stores.pinecone_store import PineconeStore

store = PineconeStore()
store.delete_index()  # Удалить индекс Pinecone
```

### Просмотр логов
```bash
# Логи сохраняются в папке logs/
type logs\rag_demo_*.log  # Windows
cat logs/rag_demo_*.log   # Linux/Mac
```

---

## Советы по использованию

1. **Начните с малого**: Сначала протестируйте на 10-20 документах
2. **Используйте метаданные**: Добавляйте source, date, category для фильтрации
3. **Экспериментируйте с top_k**: Обычно 3-5 результатов достаточно
4. **Чанкинг для больших текстов**: Разбивайте документы >1000 слов
5. **Сравнивайте БД**: Pinecone быстрее, Weaviate гибче
6. **Мониторьте затраты**: OpenAI API платный, следите за использованием

---

## Что дальше?

- Интегрируйте с LLM (GPT-4, Claude) для генерации ответов
- Добавьте веб-интерфейс (Streamlit, Gradio)
- Создайте API (FastAPI, Flask)
- Подключите к Telegram/Discord боту
- Добавьте мультимодальность (изображения, аудио)

---

## Поддержка

Если возникли вопросы:
1. Проверьте README.md
2. Запустите `python scripts/check_setup.py`
3. Посмотрите логи в папке `logs/`
4. Создайте issue на GitHub
