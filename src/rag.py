import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_db"

def get_embeddings_model():
    """
    Инициализирует модель для создания эмбеддингов (числовых векторов) из текста.
    """
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_vector_db():
    """Создает базу данных векторов из наших чанков текста."""
    from ingest import load_and_chunk_documents, fetch_and_save_articles

    print("Получаем чанки текстов...")
    fetch_and_save_articles()
    chunks = load_and_chunk_documents()

    print("Инициализируем модель эмбеддингов...")
    embeddings = get_embeddings_model()

    print("Создаем векторную базу данных Chroma...")
    # Создаем БД. Если папка уже существует, она будет перезаписана или дополнена
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"База данных успешно создана и сохранена в {CHROMA_PATH}")
    return db

def get_vector_db():
    """Загружает уже существующую базу данных."""
    embeddings = get_embeddings_model()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return db

def format_docs(docs):
    """Форматирует найденные документы в единый текст для передачи в LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_pipeline():
    """Настраивает полный RAG пайплайн: поиск (Retrieval) + генерация ответа (Generation)."""

    # 1. Загружаем базу данных и создаем "retriever"
    # Retriever — это интерфейс, который ищет похожие документы по запросу
    db = get_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Ищем топ-3 подходящих куска текста

    # 2. Инициализируем LLM (языковую модель)
    # Здесь используется OpenAI. Чтобы это работало, нужен ключ в переменной окружения OPENAI_API_KEY.
    # Для портфолио можно использовать бесплатные альтернативы (например HuggingFaceHub или локальную Llama).
    # Но для качества и простоты пока возьмем GPT-4o-mini (или GPT-3.5)

    # ПРОВЕРКА КЛЮЧА
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ВНИМАНИЕ: Не установлен OPENAI_API_KEY! Модель может не работать.")

    # Мы используем фиктивный ключ для демонстрации, если его нет
    # В реальности нужно задать: export OPENAI_API_KEY="sk-..."
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0, # 0 означает, что модель не будет фантазировать
        api_key=api_key if api_key else "dummy_key"
    )

    # 3. Создаем промпт (инструкцию для модели)
    template = """Ты — вежливый и полезный ИИ-ассистент Т-Банка.
Используй приведенные ниже фрагменты контекста, чтобы ответить на вопрос пользователя.
Если ты не знаешь ответа или ответа нет в контексте, просто скажи: "Извините, я не нашел информации по вашему вопросу в базе знаний."
Не придумывай информацию от себя.

Контекст:
{context}

Вопрос: {question}

Полезный ответ:"""

    prompt = PromptTemplate.from_template(template)

    # 4. Собираем цепь (Pipeline) с помощью LangChain Expression Language (LCEL)
    rag_chain = (
        # Сначала получаем контекст из ретривера, а вопрос пробрасываем как есть
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # Затем передаем это в промпт
        | prompt
        # Затем в языковую модель
        | llm
        # И парсим результат в обычную строку
        | StrOutputParser()
    )

    return rag_chain

def answer_question(question: str):
    """Функция для удобного тестирования RAG пайплайна."""
    chain = setup_rag_pipeline()
    print(f"Вопрос: {question}")
    print("-" * 50)

    try:
        response = chain.invoke(question)
        print(f"Ответ: {response}")
    except Exception as e:
        print(f"Произошла ошибка (возможно, нужен реальный OPENAI_API_KEY): {e}")

if __name__ == "__main__":
    print("Создаем базу данных (если еще не создана)...")
    create_vector_db()
    print("\n" + "="*50 + "\n")

    # Тестируем:
    answer_question("Сколько стоит обслуживание карты T-Black?")
    print("\n")
    answer_question("Как оформить карту?")
    print("\n")
    answer_question("Какая погода в Москве?") # Проверка на то, что бот не будет фантазировать
