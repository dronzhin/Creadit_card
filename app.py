import streamlit as st
from mypages import load_csv, data_analysis, my_autokeras, main  # Импорт дочерних страниц

selected_category = st.sidebar.radio("Меню", ["Главная", "Загрузка", "Анализ", "Автокерас"])

if selected_category == "Главная":
    main.load()
elif selected_category == "Загрузка":
    load_csv.load()
elif selected_category == "Анализ":
    data_analysis.load()
elif selected_category == "Автокерас":
    my_autokeras.load()