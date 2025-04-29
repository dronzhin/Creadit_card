import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from mypages.Genetic.my_bot import createRandomNet, model_param
from mypages.Genetic.my_model import Create_model
from mypages.Genetic.torch_dataset import Dataset

# Создание класса для работы с моделью
class My_model():

    # Параметры класса
    model = None      # Модель
    dataset = None    # Класс датасета, который хранит данные для обучения и анализа
    loss = None       # Функция ошибки
    optim = None      # Оптимизатор
    history = None    # История обучения

    # Инициализация класса
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    # Обучение модели
    def train(self, epochs=5, loss = 'BCEWithLogitsLoss', opt = 'SGD', lrs = [0.001]):

        # Проверяем возможность подключения видеокарты
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Выбираем loss
        self.loss = self.__loss(loss)

        # Запускаем обучение
        losses = []
        val_losses = []
        my_accuracy = []
        val_accuracy = []


        for lr in lrs:

            losses_lr = []
            val_losses_lr = []
            my_accuracy_lr = []
            val_accuracy_lr = []

            # Выбираем оптимизатор
            self.optim = self.__optim(opt, lr)

            for epoch in range(epochs):

                # Обучающая выборка
                self.model.train()
                lossTot = 0
                accTot = 0
                for data, target in tqdm.tqdm(self.dataset.train_loader, desc=f'Обработка {epoch + 1} эпохи'):
                    self.optim.zero_grad()
                    output = self.model(data.to(device))

                    # Подсчет ошибки
                    my_loss = self.loss(output, target.to(device))
                    lossTot += my_loss.item()

                    # Подсчет точности
                    my_acc = self.accuracy(output, target.to(device))
                    accTot += my_acc

                    my_loss.backward()
                    self.optim.step()
                losses_lr.append(lossTot/len(self.dataset.train_loader))
                my_accuracy_lr.append(accTot/len(self.dataset.train_loader))

                # Тестовая выборка
                val_totalLoss = 0
                val_accTot = 0
                with torch.no_grad():
                    for data, target in self.dataset.test_loader:
                        output = self.model(data.to(device))

                        # Подсчет ошибки
                        my_loss = self.loss(output, target.to(device))
                        val_totalLoss += my_loss.item()

                        # Подсчет точности
                        my_acc = self.accuracy(output, target.to(device))
                        val_accTot += my_acc

                    val_losses_lr.append(val_totalLoss/len(self.dataset.test_loader))
                    val_accuracy_lr.append(val_accTot/len(self.dataset.test_loader))

                print(f'Эпоха [{epoch + 1}/{epochs}], lr - {lr}, Ошибка {round(losses_lr[epoch], 6)}, Тестовая ошибка {round(val_losses_lr[epoch], 6)}, '
                      f'Точность {round(my_accuracy_lr[epoch], 6) * 100}%, Тестовая точность {round(val_accuracy_lr[epoch], 6)*100}%')
            losses.extend(losses_lr)
            val_losses.extend(val_losses_lr)
            my_accuracy.extend(my_accuracy_lr)
            val_accuracy.extend(val_accuracy_lr)

        self.history = {'losses' : losses, 'val_losses' : val_losses, 'my_accuracy' : my_accuracy, 'val_accuracy' : val_accuracy}


    # Функция выбора ошибки
    def __loss(self, loss = 'MSE'):
        # Среднеквадратическая ошибка
        if loss == 'MSE':
            return nn.MSELoss()
        # Среднеабсолютная ошибка
        elif loss == 'L1Loss':
            return nn.L1Loss()
        # Ошибка кроссэнтропии
        elif loss == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        # Ошибка для бинарной классификации
        elif loss == 'BCELoss':
            return nn.BCELoss()
        elif loss == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss()
        else:
            print('Не правильно введена функция ошибка')
            return None

    # Функция выбора оптимизатора
    def __optim(self, opt = 'SGD', lr = 0.001):
        if opt == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        elif opt == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt == 'RMS':
            return torch.optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            print('Не правильно введена функция оптимизатора')
            return None

    @staticmethod
    def accuracy(model_output, labels):
        # 1. Преобразование логитов в вероятности через сигмоиду
        probabilities = torch.sigmoid(model_output)  # [[1]]

        # 2. Получение предсказаний (0 или 1)
        predictions = (probabilities > 0.5).float()

        # 3. Сравнение с таргетами
        correct = (predictions == labels).sum().item()

        # 4. Расчет accuracy
        accuracy = correct / labels.size(0)

        return accuracy

    # Отображение графиков на обучающей и тестовой выборке
    def show_test_val(self, start=0, end=None, step=1):
        if end is None:
            end=len(self.history['losses'])
        plt.plot(self.history['losses'][start:end:step],
        label='Средняя ошибка на обучающем наборе')
        plt.plot(self.history['val_losses'][start:end:step],
        label='Средняя ошибка на проверочном наборе')
        plt.ylabel('Средняя ошибка')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    bot = createRandomNet(model_param)
    print(bot)
    model = Create_model(bot, 29)
    print(model)

    path = '/home/dmitry/PycharmProjects/Creadit_card/dataset/creditcard_2023.csv'
    test = 0.1
    batch = 128
    dataset = Dataset(path, header=0, norm=2, test=test, batch=batch)
    example_model = My_model(model, dataset)
    lrs = [0.001, 0.0005, 0.0001]
    example_model.train(lrs=lrs)
    example_model.show_test_val()
