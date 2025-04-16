import streamlit as st
import pandas as pd

def load():

    st.subheader("Загрузка файла CSV")
    num_rows = st.number_input("Количество строк для отображения", min_value=1, value=20, key='num_rows')
    uploaded_file = st.file_uploader("Загрузите CSV-файл с банковскими операциями", type=["csv"], key='uploaded_file')

    if uploaded_file is not None:
        # Чтение CSV-файла
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.subheader(f"Первые {num_rows} строк данных:")
        st.dataframe(data.head(num_rows))