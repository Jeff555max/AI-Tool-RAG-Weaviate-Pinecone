"""
RAG Generator - генерация ответов на основе найденных документов
"""

from typing import List, Dict, Any
from openai import OpenAI
from loguru import logger

from config.settings import settings


class RAGGenerator:
    """Генератор ответов на основе контекста из векторной БД"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Инициализация генератора
        
        Args:
            model: Модель OpenAI для генерации ответов
        """
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        logger.info(f"Initialized RAGGenerator with model: {model}")
    
    def generate_answer(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        language: str = "russian"
    ) -> str:
        """
        Генерация ответа на основе найденных документов
        
        Args:
            query: Вопрос пользователя
            context_documents: Список найденных документов с текстом и score
            language: Язык ответа (russian/english)
            
        Returns:
            Сгенерированный ответ
        """
        if not context_documents:
            return "Извините, я не нашел релевантной информации в базе знаний для ответа на ваш вопрос."
        
        # Формируем контекст из найденных документов
        context = "\n\n".join([
            f"Документ {i+1} (релевантность: {doc['score']:.2f}):\n{doc['text']}"
            for i, doc in enumerate(context_documents)
        ])
        
        # Системный промпт
        system_prompt = """Ты - полезный ассистент, который отвечает на вопросы ТОЛЬКО на основе предоставленного контекста.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском языке
2. Используй ТОЛЬКО информацию из предоставленных документов
3. Если в документах нет ответа на вопрос, честно скажи об этом
4. НЕ придумывай информацию, которой нет в документах
5. Давай конкретные и точные ответы
6. Ссылайся на документы при ответе (например: "Согласно документу...")"""

        # Пользовательский промпт
        user_prompt = f"""Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Ответь на вопрос на русском языке, используя ТОЛЬКО информацию из предоставленного контекста."""

        try:
            logger.info(f"Generating answer for query: '{query[:50]}...'")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Низкая температура для точности
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            logger.info("Answer generated successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Ошибка при генерации ответа: {e}"
    
    def generate_answer_with_sources(
        self,
        query: str,
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Генерация ответа с указанием источников
        
        Args:
            query: Вопрос пользователя
            context_documents: Список найденных документов
            
        Returns:
            Словарь с ответом и источниками
        """
        answer = self.generate_answer(query, context_documents)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "text": doc["text"][:200] + "...",
                    "score": doc["score"]
                }
                for doc in context_documents
            ],
            "num_sources": len(context_documents)
        }
