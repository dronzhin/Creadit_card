import torch
import streamlit as st
import matplotlib.pyplot as plt


def plot_history(history):

    # Заголовок
    st.subheader("Графики ошибок и точности")
    st.markdown("---")
    epochs = range(1, len(history['losses']) + 1)

    # Потери
    plt.figure(figsize=(12, 18))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['losses'], label='Обучение')
    plt.plot(epochs, history['val_losses'], label='Тест')
    plt.title('Потери')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()

    # Точность
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['my_accuracy'], label='Обучение')
    plt.plot(epochs, history['val_accuracy'], label='Тест')
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.ylabel('Accuracy')
    plt.legend()

    st.pyplot(plt)

    # Заголовок
    st.subheader("Метрики модели")
    st.markdown("---")

    # Визуализация метрик
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Итоговая точность", f"{history['my_accuracy'][-1]:.4%}", delta=f"{1-history['my_accuracy'][-1]:.6f}% до 100%", delta_color="normal")
    with col2:
        st.metric("Финальная ошибка", f"{history['losses'][-1]:.4f}")

    # Прогресс-бар
    st.progress(int(history['my_accuracy'][-1] * 100))
    st.caption("Визуализация уровня точности модели")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Итоговая точность", f"{history['val_accuracy'][-1]:.4%}", delta=f"{1-history['val_accuracy'][-1]:.6f}% до 100%", delta_color="normal")
    with col2:
        st.metric("Финальная ошибка", f"{history['val_losses'][-1]:.4f}")

    # Прогресс-бар с точностью
    st.progress(int(history['val_accuracy'][-1] * 100))
    st.caption("Визуализация точности на тестовой выборке")


def analyze_model_structure(model):
    """Анализирует структуру PyTorch модели и формирует заключение"""
    st.subheader("Анализ структуры")

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Сбор информации о слоях
    layers = []
    layer_types = []
    for name, module in model.named_children():
        layers.append(name)
        layer_types.append(module.__class__.__name__)

    # Формирование вывода
    conclusion = f"""
    ### Заключение по структуре модели:

    1. **Общая информация**:
       - Количество слоев: **{len(layers)}**
       - Всего параметров: **{total_params:,}**
       - Обучаемые параметры: **{trainable_params:,}** ({trainable_params / total_params:.1%})
       - Необучаемые параметры: **{non_trainable_params:,}** 

    2. **Слои модели**:
       - Список: **{', '.join(layers)}**
       - Типы: **{', '.join(layer_types)}**

    3. **Анализ**:
       - Модель использует комбинацию слоев: {', '.join(set(layer_types))}
       - Наличие необучаемых параметров ({non_trainable_params:,}) может указывать на:
         - Слои нормализации (BatchNorm)
         - Предобученные компоненты
         - Фиксированные эмбеддинги
    """

    st.markdown(conclusion)