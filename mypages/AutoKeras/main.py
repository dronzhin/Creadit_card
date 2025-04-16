from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pandas as pd
import os
import base64
from my_Keras import get_best_model
import tensorflow as tf
import tempfile

app = FastAPI()


@app.post("/Train")
async def train(
    file: UploadFile = File(None),
    trial: int = Form(10),
    start_epochs: int = Form(10),
    finish_epochs: int = Form(50)
):
    try:
        if file is not None:
        # 1. Чтение CSV-файла
            df = pd.read_csv(file.file)
        else:
            df = pd.read_csv('creditcard_2023.csv')


        # 2. Обучение модели
        result = get_best_model(df, trial, start_epochs, finish_epochs)

        # 3. Сохранение модели во временный файл
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            temp_file_path = temp_file.name
            tf.keras.models.save_model(result["best_model"], temp_file_path)

        # Чтение модели из временного файла и кодирование в Base64
        with open(temp_file_path, "rb") as f:
            model_b64 = base64.b64encode(f.read()).decode('utf-8')

        # Удаление временного файла
        os.unlink(temp_file_path)

        # 4. Формирование ответа с метаданными
        return {
            "model": model_b64,
            "eval_loss": result["eval_loss"],
            "eval_accuracy": result["eval_accuracy"],
            "history": result["history"]
        }

    except Exception as e:
        print(f"Ошибка: {str(e)}")  # Логирование для отладки
        raise HTTPException(
            status_code=500,
            detail="Ошибка при обработке запроса. Проверьте логи сервера."
        )

# Обработчик для корневого маршрута (опционально)
@app.get("/")
async def read_root():
    return {
        "message": "Добро пожаловать в API запрос AutoKeras",
        "routes": {
            "/Train": "Запрос на подбор и обучение",
        }
    }

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)


