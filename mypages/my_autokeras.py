import streamlit as st
import pickle
import autokeras as ak
import pandas as pd
from mypages.AutoKeras.autokeras_architecture import show_model_architecture, analyze_model_structure
from mypages.AutoKeras.autokeras_history import plot_training_history
from mypages.AutoKeras.autoreras_test import test_random_samples

# --- Загрузка модели и данных ---
@st.cache_resource
def load_tf_model():
    with open('mypages/AutoKeras/model_keras.pickle', 'rb') as f:
        result = pickle.load(f)
    model = result['best_model']
    return model

def load_data_sample():
    df = pd.read_csv("dataset/creditcard_2023.csv").sample(10)
    return df

def load():
    # Заголовок и структура
    st.title("Анализ лучшей модели AutoKeras")

    # Загрузка компонентов
    model = load_tf_model()
    data = load_data_sample()

    # Вкладки
    tab1, tab2, tab3 = st.tabs(["Модель", "Обучение", "Тестирование"])

    with tab1:
        show_model_architecture(model)
        # Анализ и вывод заключения
        conclusion = analyze_model_structure(model)
        st.markdown(conclusion)

    with tab2:
        plot_training_history()
    with tab3:
        test_random_samples(model, data)


