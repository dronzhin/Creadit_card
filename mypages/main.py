import streamlit as st

def load():
    # Заголовок
    st.title("📊 Многофункциональная система анализа данных для определения мошеннических операций")
    st.markdown("---")

    with st.container():
        st.header("👋 Добро пожаловать!")
        st.markdown("""
        Это приложение предназначено для автоматизации анализа данных и оптимизации моделей машинного обучения.  
        Используйте боковое меню для навигации по разделам:
        - 📁 **Загрузка датасета** — импорт и предварительный просмотр данных  
        - 🔍 **Анализ данных** — автоматическая оценка качества и характеристик  
        - 🤖 **Автокерас** — результаты автоматического машинного обучения  
        - 🧬 **Генетический подбор модели** — оптимизация гиперпараметров
        """)

        st.header("📥 Загрузка датасета")
        st.markdown("""
        #### Возможности:
        - Укажите количество строк для отображения с помощью слайдера  
        - Загрузите CSV-файл через drag-and-drop интерфейс  
        - Фрагмент датасета отобразится в табличном формате после выбора  
    
        ⚠️ Примечание: Для корректной работы данные должны быть в формате CSV с разделителем запятой.
        """)

        st.header("📊 Автоматический анализ данных")

        st.markdown("""
        - 🧮 Общий анализ
        - ✅ Полнота данных" 
        - ⚖️ Сбалансированность классов
        - ⚠️ Подозрительные значения
        """)

        st.header("🧠 Результаты AutoKeras")

        st.markdown("""
        - 🧠 Архитектура лучшей модели
        - 📉 Графики потерь и точности
        - 🧪 Интерактивное тестирование. Выбор случайных 10 записей из датасета. Предсказания сравниваются с истинными значениями в таблице.
        """)

        st.header("🧬 Оптимизация гиперпараметров")

        st.markdown("""
         - 🔍 Параметры"
         - "🏆 Лучшие модели
         - 💡 Тестирование. Выбор случайных 10 записей из датасета. Предсказания сравниваются с истинными значениями в таблице.
         """)
