import requests
import base64

url = "http://localhost:8010/Train"
file_path = "creditcard_2023.csv"

with open(file_path, "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "trial": 2,          # Передаем как отдельные поля
            "start_epochs": 2,
            "finish_epochs": 2
        }
    )
    print("Ответ сервера:", response.status_code)
    print("Тело ответа:", response.text)


# Парсинг ответа
data = response.json()

# Сохранение модели из Base64
model_bytes = base64.b64decode(data["model"])
with open("model.h5", "wb") as f:
    f.write(model_bytes)

print("Модель сохранена в model.h5")
print("Метаданные:")
print(f" - Loss: {data['eval_loss']}")
print(f" - Accuracy: {data['eval_accuracy']}")
print(f" - История обучения: {data['history']}")