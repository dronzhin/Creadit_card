import autokeras as ak
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os

def get_best_model(data: pd.DataFrame, trial =10, start_epochs = 10, finish_epochs = 50):

    # Получение обучающей и тестовой выборки
    if data is not None and not data.empty:
        x_train, x_test, y_train, y_test = create_dataset(data, 0.1)
    elif os.path.exists('data.pickle'):
        # Загрузка датасета из pickle [[9]]
        with open('data.pickle', 'rb') as f:
            data_load = pickle.load(f)
            x_train = data_load['x_train']
            x_test = data_load['x_test']
            y_train = data_load['y_train']
            y_test = data_load['y_test']
    else:
        raise ValueError("Нет данных")

    clf = ak.StructuredDataClassifier(max_trials=trial)
    clf.fit(x_train, y_train, epochs=start_epochs)

    # Экспорт и дообучение
    best_model = clf.export_model()
    history = best_model.fit(x_train, y_train, epochs=finish_epochs)

    # Оценка модели на тестовой выборке
    res_eval = best_model.evaluate(x_test, y_test)

    res = {
        "eval_loss": res_eval[0],
        'eval_accuracy': res_eval[1],
        'best_model': best_model,
        'history' : history.history
    }

    return res

def create_dataset(data: pd.DataFrame, test_size, column = None):
    # Получение абсолютного пути
    data = data.reset_index(drop=True)  # Удаляет старый индекс

    if column:
        x = data[column].to_numpy()  # Если column задан
    else:
        # Проверка наличия столбцов
        if "id" in data.columns and "Class" in data.columns:
            x = data.drop(columns=["id", "Class"]).to_numpy()
        else:
            raise ValueError("Столбцы 'id' или 'Class' отсутствуют в данных")

    y = data["Class"].to_numpy()  # y содержит только 'Class'
    # Разделение на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    # Относительный путь к файлу
    path = "creditcard_2023.csv"

    # Проверка работы автокерас
    result = get_best_model(path, trial =2, start_epochs = 3)
    print(result['history'])
