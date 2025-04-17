import pandas as pd
import streamlit as st

# Функция для стилизации строк
def highlight_row(row):
    # Сравниваем истинный и предсказанный класс в строке
    match = row["Истинный класс"] == row["Предсказанный класс"]
    return [
        "background-color: lightgreen" if match else "background-color: salmon"
        for _ in row
    ]


def test_random_samples(model, data):
    st.subheader("Тестирование на случайных данных")

    # Функция для стилизации строк
    def highlight_row(row):
        match = row["Истинный класс"] == row["Предсказанный класс"]
        return [
            "background-color: lightgreen" if match else "background-color: salmon"
            for _ in row
        ]

    if st.button("Загрузить 10 случайных записей"):
        samples = data.copy()
        st.write("Пример данных:")
        st.dataframe(samples)

        # Предобработка
        X = samples.drop(["Class", 'id'], axis=1).values
        y_true = samples["Class"].values.flatten()  # Убедимся, что y_true одномерный

        # Предсказание
        y_pred = model.predict(X)
        y_pred = (y_pred > 0.5).astype(int).flatten()  # Добавлено .flatten() [[6]]

        # Создаем DataFrame с результатами
        results = pd.DataFrame({
            "Истинный класс": y_true,
            "Предсказанный класс": y_pred
        })

        # Применяем стилизацию
        styled_results = results.style.apply(highlight_row, axis=1)

        # Выводим результаты
        st.write("Результаты классификации:")
        st.dataframe(styled_results)