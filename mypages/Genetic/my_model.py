import torch.nn as nn
from mypages.Genetic.my_bot import createRandomNet, model_param

# Создаем класс для формировании модели бота
class Create_model(nn.Module):

    # Параметры класса
    net = None                # Параметры бота

    def __init__(self, net, inp):
        super(Create_model, self).__init__()
        self.net = net
        self.out1 = 2**net[0]
        self.out2 = 2**net[7]
        self.out3 = 2**net[14]

        # Узнаем вход в последний слой
        if net[5] == 1:
            self.out = self.out2
            if net[12] == 1:
                self.out = self.out3
        else:
            self.out = self.out1

        # Первый блок
        self.line1 = nn.Linear(in_features=inp, out_features=self.out1, bias=net[1])
        if self.net[2] > 0:
            self.act1 = self.__activation(net[2])
        if self.net[3] == 1:
            self.dropout1 = nn.Dropout(p=0.1 * net[4])

        # Второй блок
        if self.net[5] == 1:
            if self.net[6] == 1: # Проверка на нормализацию
                self.bn2 = nn.BatchNorm1d(self.out1)
            self.line2 = nn.Linear(in_features=self.out1, out_features=self.out2, bias=net[8])
            if self.net[9] > 0:  # Проверка на активизацию
                self.act2 = self.__activation(net[9])
            if self.net[10] == 1:  # Проверка на дроппаут
                self.dropout2 = nn.Dropout(p=0.1 * net[11])

        # Третий блок
        if self.net[12] == 1:
            if self.net[13] == 1: # Проверка на нормализацию
                self.bn3 = nn.BatchNorm1d(self.out2)
            self.line3 = nn.Linear(in_features=self.out2, out_features=self.out3, bias=net[15])
            if self.net[16] > 0:  # Проверка на активизацию
                self.act3 = self.__activation(net[16])
            if self.net[17] == 1:  # Проверка на дроппаут
                self.dropout3 = nn.Dropout(p=0.1 * net[18])

        # Выходной блок
        self.line_end = nn.Linear(in_features=self.out, out_features=1, bias=net[15])

    # Функция определения активизации
    @staticmethod
    def __activation(num):
        if num == 1:
            return nn.ReLU()
        elif num == 2:
            return nn.Tanh()
        elif num == 3:
            return nn.LeakyReLU()
        elif num == 4:
            return nn.Sigmoid()
        else:
            return None

    def forward(self, x):

        # 1 блок
        x = self.line1(x)
        if self.net[2] > 0: # Проверка на активизацию
            x = self.act1(x)
        if self.net[3] == 1: # Проверка на дроппаут
            x = self.dropout1(x)

        # 2 блок
        if self.net[5] == 1:
            if self.net[6] == 1: # Проверка на нормализацию
                x = self.bn2(x)
            x = self.line2(x)
            if self.net[9] > 0:  # Проверка на активизацию
                x = self.act2(x)
            if self.net[10] == 1:  # Проверка на дроппаут
                x = self.dropout2(x)

        # 3 блок
        if self.net[12] == 1:
            if self.net[13] == 1: # Проверка на нормализацию
                x = self.bn3(x)
            x = self.line3(x)
            if self.net[16] > 0:  # Проверка на активизацию
                x = self.act3(x)
            if self.net[17] == 1:  # Проверка на дроппаут
                x = self.dropout3(x)

        return self.line_end(x)

if __name__ == '__main__':
    for i in range(3):
        bot = createRandomNet(model_param)
        #print(bot)
        model = Create_model(bot, 19)
        print(model)