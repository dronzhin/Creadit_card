import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Создаем класс для формирования датасета
class Dataset:

    # Инициализация класса
    def __init__(self, csv_filename, header='infer', norm=None, test = 0.2, batch=32):
        self.data = self.__load_csv_file(csv_filename, header=header)
        self.size = self.data.shape[0]
        self.len = self.data.shape[1] - 2 # Количество признаков, убираем индексы и метки
        self.__create_dataset(norm=norm)
        self.__create_dataloader(test=test, batch=batch)

    # Функция загрузки csv файла
    @staticmethod
    def __load_csv_file(csv_filename, header):
        try:
            return pd.read_csv(csv_filename, header=header)

        except FileNotFoundError:
            print(f"Ошибка: файл '{csv_filename}' не найден.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return

    def __create_dataset(self, norm = None):
        # Получение абсолютного пути
        self.data = self.data.reset_index(drop=True)  # Удаляет старый индекс

        self.data_class = self.data["Class"]
        self.data_class = self.pandas_to_torch(self.data_class).unsqueeze(1)

        if "id" in self.data.columns and "Class" in self.data.columns:
            self.data = self.data.drop(columns=["id", "Class"])
            self.data = self.pandas_to_torch(self.data, norm=norm)
        else:
            raise ValueError("Столбцы 'id' или 'Class' отсутствуют в данных")

    # Функция перевода таблицы пандас в тензор торч
    def pandas_to_torch(self, data, norm = None):
        # Преобразование DataFrame в тензор
        data = data.astype('float32')
        data_torch = torch.tensor(data.values).squeeze()

        # Нормализация данных
        # 1 - StandardScaler: Масштабирует данные путем удаления среднего и масштабирования до единичной дисперсии.
        # 2 - MinMaxScaler: Масштабирует данные в интервале [0, 1] или в произвольный заданный диапазон.
        # 3 - MaxAbsScaler: Масштабирует данные до [-1, 1] путем деления на максимальное по модулю значение.
        # 4 - RobustScaler: Масштабирует данные, учитывая выбросы и медиану.
        # 5 - QuantileTransformer: Преобразует данные так, чтобы они имели нормальное распределение.
        # 6 - PowerTransformer: Преобразует данные так, чтобы они имели более гауссово-подобное распределение.
        if norm in [1, 2, 3, 4, 5, 6]:  # Проверяем были ли введены какие-то значения нормализации
            if norm == 1:
                self.Scaler = StandardScaler()
            elif norm == 2:
                self.Scaler = MinMaxScaler()
            elif norm == 3:
                self.Scaler = MaxAbsScaler()
            elif norm == 4:
                self.Scaler = RobustScaler()
            elif norm == 5:
                self.Scaler = QuantileTransformer()
            elif norm == 6:
                self.Scaler = PowerTransformer()
            self.Scaler.fit(data_torch)
            data_torch = self.Scaler.transform(data_torch)
            data_torch = torch.tensor(data_torch).squeeze()
        elif norm != None:
            print('Нет такого параметра нормализации')
        return data_torch.float()

    def __create_dataloader(self, test, batch):
        train_size = int(self.size * (1 - test))
        train_data = TensorDataset(self.data[:train_size], self.data_class[:train_size])
        test_data = TensorDataset(self.data[train_size:], self.data_class[train_size:])
        self.train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)



if __name__ == '__main__':
    path = '/home/dmitry/PycharmProjects/Creadit_card/dataset/creditcard_2023.csv'
    test = 0.1
    batch = 2

    data = Dataset(path, header=0, norm=2, test=test, batch=batch)
    print(data.len)
    # создаем итератор
    train_iter = iter(data.train_loader)
    # в данном случае, получаем только первые 3 батча
    for _ in range(3):
        print(next(train_iter))

    # создаем итератор
    test_iter = iter(data.test_loader)
    # в данном случае, получаем только первые 3 батча
    for _ in range(3):
        print(next(test_iter))