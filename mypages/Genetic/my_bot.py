import random

# Функция для генерации бота
def createRandomNet(list):
    net = [] # Список, где будут сгенерированы

    # Проходим по кортежу переданных данных, генерируем по ним параметр и добавляем в net
    try:
        for arg in list:
            if len(arg) == 2 and all(isinstance(i, int) for i in arg): # Длина должна быть равна 2 и оба значения должны быть типом INT
                net.append(random.randint(arg[0], arg[1]))

        # Возвращаем сгенерированный массив
        if net[5] == 0:
            for i in range(6, 19):
                net[i] = 0
        elif net[12] == 0:
            for i in range(13, 19):
                net[i] = 0

        # Обноляем велечину дроппаут при его отсутствии
        net[4] = net[4] * net[3]
        net[11] = net[11] * net[10]
        net[18] = net[18] * net[17]

        return net

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return

# Создаем параметры формирования модели
model_param = [

    (3,12), #Первый выход линейного слоя от 8 до 1024 нейронов (0)
    (0,1), #Есть ли Биас (1)
    (0,4), #Есть ли функция активизация и если есть, то какая (2)
    (0,1), #Делаем ли дроппаут (3)
    (1,3), #Величина дроппаут от 10 до 30% (4)

    (0,1), # Делаем ли второй слой (5)
    (0,1), # Делаем ли нормализацию (6)
    (3,12), # Выход линейного слоя от 8 до 1024 нейронов (7)
    (0,1),  # Есть ли Биас (8)
    (0,4), #Есть ли функция активизация и если есть, то какая (9)
    (0,1), #Делаем ли дроппаут (10)
    (1,3), #Величина дроппаут от 10 до 30% (11)

    (0,1), # Делаем ли третий слой (12)
    (0,1), # Делаем ли нормализацию (13)
    (3,12), # Выход линейного слоя от 8 до 1024 нейронов (14)
    (0,1),  # Есть ли Биас (15)
    (0,4), #Есть ли функция активизация и если есть, то какая (16)
    (0,1), #Делаем ли дроппаут (17)
    (1,3), #Величина дроппаут от 10 до 30% (18)
]

if __name__ == '__main__':
    # Создаем первичную сгенерированную популяцию
    popul = [createRandomNet(model_param) for _ in range(3)]
    for pop in popul:
        print(pop)