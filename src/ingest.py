import os
import asyncio
import aiohttp
import html2text
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR = "data"
URLS_FILE = "urls.txt"

def load_urls_from_file(filepath=URLS_FILE):
    """Считывает ссылки из файла, игнорируя комментарии и пустые строки."""
    urls = []
    if not os.path.exists(filepath):
        print(f"Файл {filepath} не найден. Создайте его и добавьте ссылки.")
        return urls

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls

async def fetch_url(session, url, converter):
    """Асинхронно скачивает страницу и преобразует HTML в чистый текст."""
    try:
        # User-Agent часто обязателен, иначе сайты могут вернуть 403 Forbidden
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status == 200:
                html = await response.text()
                # Преобразуем HTML в Markdown/Text
                text = converter.handle(html)

                # Сохраняем результат в файл
                filename = url.strip('/').replace('https://', '').replace('http://', '').replace('/', '_') + ".txt"
                filepath = os.path.join(DATA_DIR, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                return filepath
            else:
                print(f"Ошибка {response.status} при парсинге {url}")
                return None
    except Exception as e:
        print(f"Исключение при парсинге {url}: {e}")
        return None

async def fetch_all_urls(urls):
    """Запускает параллельное скачивание всех ссылок."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Настраиваем конвертер html2text для чистого извлечения текста (без картинок, ссылок и тд)
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_tables = False
    converter.bypass_tables = False

    saved_files = []

    # Используем aiohttp для параллельных запросов
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_url(session, url, converter))

        print(f"Начинаем параллельное скачивание {len(urls)} страниц...")
        # Собираем результаты
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                saved_files.append(res)

    print(f"Успешно скачано {len(saved_files)} страниц.")
    return saved_files

def fetch_and_save_articles():
    """Синхронная обертка для асинхронного парсинга."""
    urls = load_urls_from_file()
    if not urls:
        print("Нет ссылок для парсинга.")
        return []

    return asyncio.run(fetch_all_urls(urls))

def load_and_chunk_documents(data_dir=DATA_DIR):
    """Загружает сохраненные тексты и разбивает их на чанки."""
    documents = []

    if not os.path.exists(data_dir):
        print("Папка с данными пуста. Сначала запустите парсинг.")
        return []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())

    print(f"Загружено {len(documents)} документов из локального хранилища.")

    # Для веб-страниц хорошо подходит разделитель по абзацам
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Документы разбиты на {len(chunks)} чанков.")

    return chunks

if __name__ == "__main__":
    print("=== Старт парсинга ===")
    fetch_and_save_articles()

    print("\n=== Старт разбивки на чанки ===")
    chunks = load_and_chunk_documents()

    if chunks:
        print("\nПример первого чанка:")
        print("-" * 50)
        print(chunks[0].page_content)
        print("-" * 50)
