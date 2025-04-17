import streamlit as st

# --- Визуализация архитектуры модели ---
def show_model_architecture(model):
    st.subheader("Архитектура модели")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.code("\n".join(model_summary), language="python")


# Функция для анализа структуры модели и создания заключения
def analyze_model_structure(model):
    # Получение информации о модели
    total_params = model.count_params()
    trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    layers = [layer.name for layer in model.layers]
    layer_types = [layer.__class__.__name__ for layer in model.layers]

    # Формирование заключения
    conclusion = f"""
    ### Заключение по структуре модели:

    1. **Общая информация:**
       - Модель содержит **{len(layers)} слоев**.
       - Общее количество параметров: **{total_params}**.
       - Обучаемые параметры: **{trainable_params}**.
       - Необучаемые параметры: **{non_trainable_params}**.

    2. **Слои модели:**
       - Список слоев: **{', '.join(layers)}**.
       - Типы слоев: **{', '.join(layer_types)}**.

    3. **Анализ:**
       - Модель использует комбинацию слоев для обработки данных, включая {', '.join(set(layer_types))}.
       - Наличие необучаемых параметров ({non_trainable_params}) может указывать на использование слоев нормализации или предобученных компонентов.
    """
    return conclusion