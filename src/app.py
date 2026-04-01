import streamlit as st
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Импортируем нашу функцию RAG (используем кэширование Streamlit, чтобы не загружать модели каждый раз)
from rag import setup_rag_pipeline

st.set_page_config(page_title="Ассистент Т-Банка", page_icon="🏦", layout="centered")

@st.cache_resource
def get_chain():
    """Кэшируем пайплайн, чтобы не создавать его заново при каждом сообщении"""
    return setup_rag_pipeline()

def main():
    st.title("🏦 RAG Ассистент Т-Банка")
    st.markdown("Этот бот отвечает на вопросы по дебетовой карте T-Black, основываясь на базе знаний (FAQ).")

    # Проверка API ключа
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("ВНИМАНИЕ: Не установлен `GROQ_API_KEY`. Ответы генерироваться не будут. Пожалуйста, добавьте его в файл `.env` или укажите ниже.")
        user_api_key = st.text_input("Введите ваш Groq API Key:", type="password")
        if user_api_key:
            os.environ["GROQ_API_KEY"] = user_api_key
            st.success("Ключ установлен для этой сессии! Можете задавать вопросы.")
        else:
            st.stop()

    try:
        chain = get_chain()
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()

    # Инициализация истории сообщений в сессии (чтобы чат сохранялся при обновлении страницы)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Здравствуйте! Чем могу помочь с дебетовой картой T-Black?"}
        ]

    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ввод пользователя
    if prompt := st.chat_input("Спросите что-нибудь о T-Black (например, 'какая комиссия за снятие наличных?'):"):

        # Показываем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Показываем заглушку, пока думаем
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔄 Ищу информацию...")

            try:
                # Запускаем наш RAG
                response = chain.invoke(prompt)
                message_placeholder.markdown(response)

                # Добавляем ответ бота в историю
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Произошла ошибка при генерации ответа: {e}"
                message_placeholder.markdown(f"❌ {error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
