import time
import random
from mypages.Genetic.my_bot import createRandomNet, model_param
from mypages.Genetic.my_model import Create_model
from torch_dataset import Dataset
from model_operation import My_model
import pickle

# Функция проверка ботов, необходима, чтобы не проверять выживших ботов и дублированных, так как они уже проверены
def gen_train(popul,              # Первоначальная популяция
              dataset,            # Датасет для обучения
              model_param,        # Параметры модели
              mut,                # Вероятность мутации
              its = 5,           # Количество итераций,
              n = 20,             # Количество ботов
              nsurv = 10,         # Количество выживших (столько лучших переходит в новую популяцию)
              epochs=5, loss = 'MSE', opt = 'SGD', lr = 0.001, # Параметры для обучения
              ):

    # Необходимые переменные
    nnew = n - nsurv              # Количество новых (столько новых ботов создается)
    val = []                      # Список результатов ботов
    start = 0                     # С какого бота начинаем обучение
    checkbot = []                 # Список проверенных ботов

    # Запускае обучение генетики
    for it in range(its):
        curr_time = time.time()

        # Проверяем, первая итерация или нет, если первая то список результатов пустой и старт с 0
        if it > 0:
            val = sval[:nsurv]
            start = nsurv

        for i in range(start, n):

            # Создаем модель по боту
            bot = popul[i]
            print(f'{it+1} этерация - {i+1} бот:{bot}')

            # Проверяем, есть ли в списке проверенных ботов этот бот, если нет, то проверяем его
            if bot not in checkbot:
                model = Create_model(bot, dataset.len)

                # Создаем класс модели для обучения и анализа
                my_model = My_model(model, dataset)

                # for i in [0.001, 0.0005, 0.0002, 0.0001]:
                #     my_model.train(epochs=5, loss=loss, opt=opt, lr=i)

                # Проводим обучение модели на 5 эпохах
                my_model.train(epochs=epochs, loss = loss, opt = opt, lr = lr)

                # Добавим точность текущего бота
                val.append(min(my_model.history['val_losses']))

                # Добавляем бот в список проверенных
                checkbot.append(bot)
            else:
                print(f"Бот {bot} уже проверен")
                val.append(10000)

        # Отсортируем val и выведем 5 лучших ботов
        sval = sorted(val, reverse=0)

        # Создадим новую популяцию и добавим лучших ботов
        newpopul = []
        for i in range(nsurv):
            index = val.index(sval[i])
            newpopul.append(popul[index])

        # Добавим новую популяцию
        for i in range(nnew):
            indexp1 = random.randint(0,nsurv-1)
            indexp2 = random.randint(0,nsurv-1)
            botp1 = newpopul[indexp1]
            botp2 = newpopul[indexp2]
            newbot = []
            # Скрещиваем ботов и добавляем мутацию
            for j in range(len(model_param)):
                x = 0
                pindex = random.random()
                if pindex < 0.5:
                    x = botp1[j]
                else:
                    x = botp2[j]

                if (random.random() < mut):
                    x = random.randint(model_param[j][0], model_param[j][1])
                newbot.append(x)
            # Обнуляем не нужные значение
            if newbot[5] == 0:
                for i in range(6, 19):
                    newbot[i] = 0
            elif newbot[12] == 0:
                for i in range(13, 19):
                    newbot[i] = 0

            # Обноляем велечину дроппаут при его отсутствии
            newbot[4] = newbot[4] * newbot[3]
            newbot[11] = newbot[11] * newbot[10]
            newbot[18] = newbot[18] * newbot[17]

            # Добовляем нового боты
            newpopul.append(newbot)

        # Переписываем пополяцию для следующей итерации
        popul = newpopul

        # Количество выводимых ботов, но не больше 5
        if nsurv > 5:
            nview = 5
        else:
            nview = nsurv

        print(it, time.time() - curr_time, " ", sval[:nview],popul[:nview])
    return popul[:nsurv], sval[:nsurv]

if __name__ == '__main__':

    # Параметры генетики
    n = 50  # Общее число ботов
    nsurv = 15  # Количество выживших (столько лучших переходит в новую популяцию)
    its = 10 # Количество итераций

    # Популяция ботов
    popul = [createRandomNet(model_param) for _ in range(n)]
    # popul[1] = [12, 0, 0, 0, 0, 1, 0, 12, 0, 0, 0, 0, 1, 0, 12, 0, 0, 0, 0]
    # popul[2] = [11, 0, 0, 0, 0, 1, 0, 11, 0, 0, 0, 0, 1, 0, 11, 0, 0, 0, 0]
    # popul[0] = [10, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, 0]

    # Формирование датасета
    path = '/home/dmitry/PycharmProjects/Creadit_card/dataset/creditcard_2023.csv'
    test = 0.1
    batch = 128
    my_dataset = Dataset(path, header=0, norm=2, test=test, batch=batch)

    bots, vals = gen_train(popul=popul,  # Первоначальная популяция
                           dataset=my_dataset,  # Датасет для обучения
                           model_param=model_param,  # Параметры модели
                           mut=0.09,  # Вероятность мутации
                           its=its,  # Количество итераций,
                           n=n,  # Количество ботов
                           nsurv=nsurv,  # Количество выживших (столько лучших переходит в новую популяцию)
                           epochs=10, loss='BCEWithLogitsLoss', opt='SGD', lr=0.001,  # Параметры для обучения
                           )

    result = {
        'bots' : bots,
        'vals' : vals
    }

    with open('result_gen3.pickle', 'wb') as f:
        pickle.dump(result, f)