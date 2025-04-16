import streamlit as st
from mypages import load_csv, data_analysis, autokeras, main  # Импорт дочерних страниц

selected_category = st.sidebar.radio("Меню", ["Главная", "Загрузка", "Анализ"])

if selected_category == "Главная":
    main.load()
elif selected_category == "Загрузка":
    load_csv.load()
elif selected_category == "Анализ":
    data_analysis.load()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Если файл загружен
# if uploaded_file is not None:
#     # Чтение CSV-файла
#     df = pd.read_csv(uploaded_file)
#
#     # Создание вкладок
#     tab1, tab2 = st.tabs(["Общий анализ", "Анализ распределения"])
#
#     with tab1:
#         # Общий анализ: вывод первых N строк и статистики
#         st.write(f"Первые {num_rows} строк данных:")
#         st.dataframe(df.head(num_rows))
#
#         st.write("Статистика по данным:")
#         st.write(df.describe())
#
#         # Пример: поиск подозрительных операций (зависит от структуры данных)
#         if "amount" in df.columns:
#             suspicious_transactions = df[df["amount"] > df["amount"].quantile(0.99)]
#             st.write("Подозрительные операции (верхние 1% по сумме):")
#             st.dataframe(suspicious_transactions)
#
#     with tab2:
#         # Анализ распределения: графики
#         st.write("Анализ распределения данных")
#
#         # График 1: Heatmap пропущенных значений
#         st.write("### Heatmap пропущенных значений")
#         fig1, ax1 = plt.subplots(figsize=(20, 12))  # Создаем область для графика
#         sns_heatmap = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')  # Визуализация пропусков
#         st.pyplot(fig1)  # Отображение графика
#
#         # График 2: Распределение суммы операций (если есть столбец 'amount')
#         if "amount" in df.columns:
#             st.write("### Распределение суммы операций")
#             fig2, ax2 = plt.subplots(figsize=(10, 6))
#             sns.histplot(df["amount"], bins=50, kde=True, color='blue')
#             plt.xlabel("Сумма операции")
#             plt.ylabel("Частота")
#             st.pyplot(fig2)
#
#         # График 3: Проверка сбалансированности данных (если есть столбец 'Class')
#         if "Class" in df.columns:
#             st.write("### Проверка сбалансированности данных")
#             fig3, ax3 = plt.subplots(figsize=(10, 8))  # Создаем область для графика
#             plt.title('Проверка сбалансированности данных', fontsize=16)  # Название графика
#             sns.countplot(x='Class', data=df, ax=ax3)  # Построение countplot
#             plt.xlabel("Класс (0: Нормальная операция, 1: Мошенническая операция)")
#             plt.ylabel("Количество операций")
#             st.pyplot(fig3)
#
# else:
#     st.write("Пожалуйста, загрузите CSV-файл.")