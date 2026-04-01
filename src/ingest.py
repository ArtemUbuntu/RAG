import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Пример нескольких статей из FAQ
URLS_TO_PARSE = [
    "https://www.tbank.ru/bank/help/debit-cards/t-black/about-card/what-is-it/",
    "https://www.tbank.ru/bank/help/debit-cards/t-black/tariffs/"
]

DATA_DIR = "data"

# Пример нескольких статей из FAQ
URLS_TO_PARSE = [
    "https://www.tbank.ru/bank/help/debit-cards/t-black/about-card/what-is-it/",
    "https://www.tbank.ru/bank/help/debit-cards/t-black/tariffs/"
]

# Поскольку Т-Банк может блокировать прямые запросы или возвращать 404 на старые URL,
# для демонстрации и надежности мы замокаем несколько статей текстом.
# В реальном проекте вы бы использовали Selenium, Playwright или официальное API (если есть)

MOCK_ARTICLES = {
    "what-is-t-black.txt": """
# Что такое дебетовая карта T-Black (Т-Банк)

T-Black — это наша лучшая дебетовая карта с кэшбэком и процентом на остаток.
Ей можно расплачиваться в магазинах и интернете, снимать наличные, делать переводы и получать за это бонусы.

Главные преимущества:
Кэшбэк рублями до 30% у партнеров.
Кэшбэк до 15% в выбранных категориях каждый месяц.
1% кэшбэка за любые другие покупки.
Бесплатное снятие наличных в банкоматах Т-Банка (до 500 000 ₽ за расчетный период).
Бесплатное снятие от 3000 ₽ в любых банкоматах по всему миру.
Процент на остаток до 5% годовых с подпиской Pro.
Бесплатные переводы на карты других банков через Систему быстрых платежей (СБП).

Как оформить карту?
Оформить карту можно онлайн на нашем сайте или в приложении. Мы бесплатно доставим ее вам домой или в офис в удобное время.
""",
    "tariffs.txt": """
# Тарифы по дебетовой карте T-Black

Сколько стоит обслуживание карты?
Обслуживание карты бесплатно, если:
- на ваших счетах, вкладах и инвестициях каждый день суммарно лежит от 50 000 ₽;
- вы взяли кредит наличными в Т-Банке;
- у вас есть подписка Pro или Premium.
В остальных случаях обслуживание стоит 99 ₽ в месяц.

Оповещения об операциях
Уведомления об операциях (СМС или пуши) стоят 99 ₽ в месяц. Если у вас есть подписка Pro или Premium, то уведомления бесплатны.

Комиссия за снятие наличных
В банкоматах Т-Банка — бесплатно до 500 000 ₽ за расчетный период, далее комиссия 2% (минимум 90 ₽).
В сторонних банкоматах — бесплатно при снятии от 3000 ₽ до 100 000 ₽ за расчетный период. При снятии меньше 3000 ₽ комиссия составит 90 ₽. При превышении лимита в 100 000 ₽ комиссия составит 2% (минимум 90 ₽).
"""
}

def fetch_and_save_articles():
    """Вместо реального скачивания (которое может упасть из-за защиты сайта), сохраняем моковые данные."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    saved_files = []

    for filename, content in MOCK_ARTICLES.items():
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())

        saved_files.append(filepath)
        print(f"Сохранено: {filepath}")

    return saved_files

def load_and_chunk_documents(data_dir=DATA_DIR):
    """Загружает сохраненные тексты и разбивает их на чанки."""
    documents = []

    # Читаем все txt файлы из папки data
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())

    print(f"Загружено {len(documents)} документов.")

    # Разбиваем текст на куски (чанки)
    # chunk_size - размер куска в символах
    # chunk_overlap - перекрытие кусков, чтобы не потерять контекст на стыке
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Документы разбиты на {len(chunks)} чанков.")

    return chunks

if __name__ == "__main__":
    print("Начинаем сбор данных...")
    fetch_and_save_articles()

    print("\nРазбиваем на чанки...")
    chunks = load_and_chunk_documents()

    if chunks:
        print("\nПример первого чанка:")
        print("-" * 50)
        print(chunks[0].page_content)
        print("-" * 50)
