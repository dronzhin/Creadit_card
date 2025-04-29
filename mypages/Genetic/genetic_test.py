import pandas as pd
import streamlit as st
import pickle
from mypages.my_autokeras import load_data_sample
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from mypages.Genetic.my_model import Create_model


# Функция для стилизации строк
def highlight_row(row):
    # Сравниваем истинный и предсказанный класс в строке
    match = row["Истинный класс"] == row["Предсказанный класс"]
    return [
        "background-color: lightgreen" if match else "background-color: salmon"
        for _ in row
    ]

def pandas_to_torch(data, Scaler = None, norm = None):
    # Преобразование DataFrame в тензор
    data = data.astype('float32')
    data_torch = torch.tensor(data).squeeze()
    if norm:
        data_torch = Scaler.transform(data_torch)
    data_torch = torch.tensor(data_torch).squeeze()

    return data_torch.float()


def test_random_samples():

    st.subheader("Тестирование на случайных данных")

    # Функция для стилизации строк
    def highlight_row(row):
        match = row["Истинный класс"] == row["Предсказанный класс"]
        return [
            "background-color: lightgreen" if match else "background-color: salmon"
            for _ in row
        ]


    if st.button("Загрузить 10 случайных записей"):
        data = load_data_sample()
        samples = data.copy()
        st.write("Пример данных:")
        st.dataframe(samples)

        # Загружаем модель
        bot = [12, 0, 0, 0, 0, 1, 0, 3, 1, 3, 0, 0, 1, 0, 12, 1, 0, 0, 0]
        model = Create_model(bot, 29)  # создайте экземпляр модели
        model.load_state_dict(torch.load('mypages/Genetic/model.pth'))
        model.eval()

        # Загружаем скалер
        with open(f'mypages/Genetic/Scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)

        # Предобработка
        X = samples.drop(["Class", 'id'], axis=1).values
        y_true = samples["Class"].values.flatten()  # Убедимся, что y_true одномерный

        X = pandas_to_torch(X, Scaler=scaler, norm=True)
        y_true = pandas_to_torch(y_true).to(torch.int)

        # Предсказание
        with torch.no_grad():
            y_pred = model(X)
            y_pred = (y_pred > 0.5).to(torch.int).flatten()  # Добавлено .flatten()

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