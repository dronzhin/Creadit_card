import streamlit as st
from mypages import load_csv, data_analysis, my_autokeras, main, gen_models  # Импорт дочерних страниц
import sys
import os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

selected_category = st.sidebar.radio("Меню", ["Главная", "Загрузка", "Анализ", "Автокерас", "Генетический подбор модели"])

if selected_category == "Главная":
    main.load()
elif selected_category == "Загрузка":
    load_csv.load()
elif selected_category == "Анализ":
    data_analysis.load()
elif selected_category == "Автокерас":
    my_autokeras.load()
elif selected_category == "Генетический подбор модели":
    gen_models.load()