import streamlit as st
from mypages.Genetic.torch_archtecture import analyze_model_structure, plot_history
from mypages.Genetic.model_operation import Create_model, My_model
import pickle
from mypages.Genetic.genetic_test import test_random_samples


def load():
    tab_params, tab_models, tab_test = st.tabs([
        "🔍 Параметры",
        "🏆 Лучшие модели",
        "💡 Тестирование"
    ])

    with tab_params:
        # Разделяем параметры по слоям
        layers = ["Первый слой 🧠", "Второй слой 🧠", "Третий слой 🧠"]

        # Словарь для расшифровки параметров
        param_decoder = [
            "включить слой (0=Нет, 1=Да)",
            "нормализация (0=Нет, 1=Да)",
            "нейронов (2ⁿ: 2^3 – 2^12 (8–4096 нейронов))",
            "bias (0=Нет, 1=Да)",
            "активация (0=None, 1=ReLU, 2=Tanh, 3=LeakyReLU, 4=Sigmoid)",
            "dropout (0=Нет, 1=Да)",
            "величина dropout (10-30%)"
        ]

        st.title("Параметры нейросетевой модели 🤖")

        for layer in layers:
            st.subheader(layer)
            if layer == "Первый слой 🧠":
                for i in range(2,7):
                    description = param_decoder[i]
                    st.text(f"• {description}")
            else:
                for i in range(0,7):
                    description = param_decoder[i]
                    st.text(f"• {description}")

            st.write("---")

    with tab_models:
        st.header("Топ-3 модели с т.з. генетического подбора")
        col1, col2, col3 = st.tabs(["Первая модель 🏆", "Вторая модель 🏅", "Третья модель 🥉"])

        with open(f'mypages/Genetic/history_models.pickle', 'rb') as f:
            history = pickle.load(f)

        with col1:
            bot = [8, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 12, 0, 0, 0, 0]
            model = Create_model(bot, 29)
            st.code(model)
            # Анализ и вывод заключения
            analyze_model_structure(model)
            plot_history(history[0])

        with col2:
            bot = [11, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, 0]
            model = Create_model(bot, 29)
            st.code(model)
            # Анализ и вывод заключения
            analyze_model_structure(model)
            plot_history(history[1])

        with col3:
            bot = [11, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 10, 0, 2, 0, 0]
            model = Create_model(bot, 29)
            st.code(model)
            # Анализ и вывод заключения
            analyze_model_structure(model)
            plot_history(history[2])

    with tab_test:
        test_random_samples()