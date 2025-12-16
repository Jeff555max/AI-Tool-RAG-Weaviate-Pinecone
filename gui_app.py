"""
GUI приложение для RAG системы на PyQt6
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QComboBox, QSpinBox, QTabWidget,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox, QListWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor

from embeddings.embedder import Embedder
from rag.retriever import Retriever
from rag.generator import RAGGenerator
from utils.chunker import TextChunker


class WorkerThread(QThread):
    """Поток для выполнения операций в фоне"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class RAGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.embedder = None
        self.retriever = None
        self.generator = None
        self.documents = []
        self.init_ui()
        self.init_rag()
    
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("RAG Vector Store - AI Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Заголовок
        title = QLabel("RAG Vector Store System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Вкладки
        tabs = QTabWidget()
        tabs.addTab(self.create_documents_tab(), "Документы")
        tabs.addTab(self.create_search_tab(), "Поиск")
        tabs.addTab(self.create_compare_tab(), "Сравнение")
        main_layout.addWidget(tabs)
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
    
    def create_documents_tab(self):
        """Вкладка для работы с документами"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа: Добавление документов
        add_group = QGroupBox("Добавить документы")
        add_layout = QVBoxLayout()
        
        # Текстовое поле для ввода
        self.doc_input = QTextEdit()
        self.doc_input.setPlaceholderText("Введите текст документа или загрузите из файла...")
        self.doc_input.setMaximumHeight(200)
        add_layout.addWidget(self.doc_input)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        
        self.btn_load_file = QPushButton("Загрузить из файла")
        self.btn_load_file.clicked.connect(self.load_file)
        btn_layout.addWidget(self.btn_load_file)
        
        self.btn_add_doc = QPushButton("Добавить документ")
        self.btn_add_doc.clicked.connect(self.add_document)
        btn_layout.addWidget(self.btn_add_doc)
        
        self.btn_clear_input = QPushButton("Очистить")
        self.btn_clear_input.clicked.connect(lambda: self.doc_input.clear())
        btn_layout.addWidget(self.btn_clear_input)
        
        add_layout.addLayout(btn_layout)
        
        # Выбор векторной БД
        db_layout = QHBoxLayout()
        db_layout.addWidget(QLabel("Векторная БД:"))
        self.db_combo = QComboBox()
        self.db_combo.addItems(["pinecone", "weaviate"])
        db_layout.addWidget(self.db_combo)
        db_layout.addStretch()
        add_layout.addLayout(db_layout)
        
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)
        
        # Группа: Список документов
        list_group = QGroupBox("Добавленные документы")
        list_layout = QVBoxLayout()
        
        self.doc_list = QListWidget()
        list_layout.addWidget(self.doc_list)
        
        btn_clear_list = QPushButton("Очистить список")
        btn_clear_list.clicked.connect(self.clear_documents)
        list_layout.addWidget(btn_clear_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return widget
    
    def create_search_tab(self):
        """Вкладка для поиска"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа: Поисковый запрос
        search_group = QGroupBox("Поисковый запрос")
        search_layout = QVBoxLayout()
        
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Введите ваш вопрос...")
        self.query_input.setMaximumHeight(100)
        search_layout.addWidget(self.query_input)
        
        # Настройки поиска
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Векторная БД:"))
        self.search_db_combo = QComboBox()
        self.search_db_combo.addItems(["pinecone", "weaviate"])
        settings_layout.addWidget(self.search_db_combo)
        
        settings_layout.addWidget(QLabel("Количество результатов:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setMinimum(1)
        self.top_k_spin.setMaximum(10)
        self.top_k_spin.setValue(3)
        settings_layout.addWidget(self.top_k_spin)
        
        settings_layout.addStretch()
        search_layout.addLayout(settings_layout)
        
        # Кнопка поиска
        self.btn_search = QPushButton("Поиск")
        self.btn_search.clicked.connect(self.search)
        self.btn_search.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        search_layout.addWidget(self.btn_search)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Группа: Результаты
        results_group = QGroupBox("Результаты поиска")
        results_layout = QVBoxLayout()
        
        self.results_output = QTextEdit()
        self.results_output.setReadOnly(True)
        results_layout.addWidget(self.results_output)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        return widget
    
    def create_compare_tab(self):
        """Вкладка для сравнения БД"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа: Запрос для сравнения
        compare_group = QGroupBox("Сравнение векторных БД")
        compare_layout = QVBoxLayout()
        
        self.compare_query_input = QTextEdit()
        self.compare_query_input.setPlaceholderText("Введите запрос для сравнения результатов...")
        self.compare_query_input.setMaximumHeight(100)
        compare_layout.addWidget(self.compare_query_input)
        
        # Настройки
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Количество результатов:"))
        self.compare_top_k_spin = QSpinBox()
        self.compare_top_k_spin.setMinimum(1)
        self.compare_top_k_spin.setMaximum(10)
        self.compare_top_k_spin.setValue(3)
        settings_layout.addWidget(self.compare_top_k_spin)
        settings_layout.addStretch()
        compare_layout.addLayout(settings_layout)
        
        # Кнопка сравнения
        self.btn_compare = QPushButton("Сравнить")
        self.btn_compare.clicked.connect(self.compare_stores)
        self.btn_compare.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        compare_layout.addWidget(self.btn_compare)
        
        compare_group.setLayout(compare_layout)
        layout.addWidget(compare_group)
        
        # Группа: Результаты сравнения
        compare_results_group = QGroupBox("Результаты сравнения")
        compare_results_layout = QVBoxLayout()
        
        self.compare_output = QTextEdit()
        self.compare_output.setReadOnly(True)
        compare_results_layout.addWidget(self.compare_output)
        
        compare_results_group.setLayout(compare_results_layout)
        layout.addWidget(compare_results_group)
        
        return widget
    
    def init_rag(self):
        """Инициализация RAG компонентов"""
        try:
            self.statusBar().showMessage("Инициализация RAG системы...")
            self.embedder = Embedder()
            self.retriever = Retriever(self.embedder)
            self.generator = RAGGenerator()
            self.statusBar().showMessage("RAG система готова к работе")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось инициализировать RAG систему:\n{e}")
            self.statusBar().showMessage("Ошибка инициализации")
    
    def load_file(self):
        """Загрузка документа из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.doc_input.setPlainText(content)
                self.statusBar().showMessage(f"Файл загружен: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")
    
    def add_document(self):
        """Добавление документа в векторную БД"""
        text = self.doc_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Предупреждение", "Введите текст документа")
            return
        
        store_type = self.db_combo.currentText()
        
        # Отключаем кнопки
        self.btn_add_doc.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.statusBar().showMessage(f"Добавление документа в {store_type}...")
        
        # Создаем поток
        def add_doc():
            self.retriever.add_documents(
                texts=[text],
                store_type=store_type,
                metadata=[{"doc_id": len(self.documents)}]
            )
            return text
        
        thread = WorkerThread(add_doc)
        thread.finished.connect(self.on_document_added)
        thread.error.connect(self.on_error)
        thread.start()
        self.current_thread = thread
    
    def on_document_added(self, text):
        """Обработка успешного добавления документа"""
        self.documents.append(text)
        preview = text[:100] + "..." if len(text) > 100 else text
        self.doc_list.addItem(f"[{len(self.documents)}] {preview}")
        self.doc_input.clear()
        
        self.btn_add_doc.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Документ добавлен! Всего: {len(self.documents)}")
        
        QMessageBox.information(self, "Успех", "Документ успешно добавлен в векторную БД")
    
    def clear_documents(self):
        """Очистка списка документов"""
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Очистить список документов? (Документы останутся в векторной БД)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.documents.clear()
            self.doc_list.clear()
            self.statusBar().showMessage("Список документов очищен")
    
    def search(self):
        """Выполнение поиска"""
        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Предупреждение", "Введите поисковый запрос")
            return
        
        store_type = self.search_db_combo.currentText()
        top_k = self.top_k_spin.value()
        
        # Отключаем кнопку
        self.btn_search.setEnabled(False)
        self.statusBar().showMessage(f"Поиск в {store_type}...")
        self.results_output.clear()
        self.results_output.append("Выполняется поиск...\n")
        
        # Создаем поток
        def do_search():
            return self.retriever.retrieve(query, store_type, top_k)
        
        thread = WorkerThread(do_search)
        thread.finished.connect(lambda results: self.on_search_complete(results, query))
        thread.error.connect(self.on_error)
        thread.start()
        self.current_thread = thread
    
    def on_search_complete(self, results, query):
        """Обработка результатов поиска"""
        self.results_output.clear()
        
        self.results_output.append(f"Вопрос: {query}\n")
        self.results_output.append("=" * 80 + "\n\n")
        
        if not results:
            self.results_output.append("Результаты не найдены")
        else:
            # Генерируем ответ на основе найденных документов
            self.results_output.append("ОТВЕТ:\n")
            self.results_output.append("-" * 80 + "\n")
            
            answer = self.generator.generate_answer(query, results)
            self.results_output.append(f"{answer}\n")
            self.results_output.append("-" * 80 + "\n\n")
            
            # Показываем источники
            self.results_output.append(f"\nИСТОЧНИКИ ({len(results)} документов):\n")
            self.results_output.append("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                self.results_output.append(f"Документ {i} [Релевантность: {result['score']:.4f}]:\n")
                self.results_output.append(f"{result['text'][:200]}...\n")
                self.results_output.append("-" * 80 + "\n\n")
        
        self.btn_search.setEnabled(True)
        self.statusBar().showMessage(f"Ответ сгенерирован на основе {len(results)} документов")
    
    def compare_stores(self):
        """Сравнение векторных БД"""
        query = self.compare_query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Предупреждение", "Введите запрос для сравнения")
            return
        
        top_k = self.compare_top_k_spin.value()
        
        # Отключаем кнопку
        self.btn_compare.setEnabled(False)
        self.statusBar().showMessage("Сравнение векторных БД...")
        self.compare_output.clear()
        self.compare_output.append("Выполняется сравнение...\n")
        
        # Создаем поток
        def do_compare():
            results = {}
            for store in ["pinecone", "weaviate"]:
                try:
                    results[store] = self.retriever.retrieve(query, store, top_k)
                except Exception as e:
                    results[store] = f"Error: {e}"
            return results
        
        thread = WorkerThread(do_compare)
        thread.finished.connect(lambda results: self.on_compare_complete(results, query))
        thread.error.connect(self.on_error)
        thread.start()
        self.current_thread = thread
    
    def on_compare_complete(self, results, query):
        """Обработка результатов сравнения"""
        self.compare_output.clear()
        
        self.compare_output.append(f"Запрос: {query}\n")
        self.compare_output.append("=" * 80 + "\n\n")
        
        for store, store_results in results.items():
            self.compare_output.append(f"=== {store.upper()} ===\n")
            
            if isinstance(store_results, str):
                self.compare_output.append(f"{store_results}\n\n")
            elif not store_results:
                self.compare_output.append("Результаты не найдены\n\n")
            else:
                for i, result in enumerate(store_results, 1):
                    self.compare_output.append(f"{i}. [Score: {result['score']:.4f}]")
                    self.compare_output.append(f"   {result['text'][:100]}...\n")
                self.compare_output.append("\n")
        
        self.btn_compare.setEnabled(True)
        self.statusBar().showMessage("Сравнение завершено")
    
    def on_error(self, error_msg):
        """Обработка ошибок"""
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_msg}")
        self.statusBar().showMessage("Ошибка выполнения операции")
        
        # Включаем кнопки обратно
        self.btn_add_doc.setEnabled(True)
        self.btn_search.setEnabled(True)
        self.btn_compare.setEnabled(True)
        self.progress_bar.setVisible(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    window = RAGApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
