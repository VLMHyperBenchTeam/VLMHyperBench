import os
import time

import torch
from model_interface.model_factory import ModelFactory
from some_package.my_module import Calculator, greet
from tqdm import tqdm

if __name__ == "__main__":
    # Имя файла, в который будем записывать данные
    filename = "data/task1_output.txt"

    # Тест модуля добавленного в докер-контейнер
    print("Тестируем some_package")
    print("-" * 30)
    print(greet(name="Ivan"))
    calc = Calculator()
    print(calc.add(1, 3))
    print("-" * 30)

    os.system("nvidia-smi")
    print("torch.__version__", torch.__version__)
    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.current_device()", torch.cuda.current_device())
    print("torch.cuda.device_count()", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

    with open(filename, "w") as file:
        for i in tqdm(range(5), desc="Processing", unit="iteration"):
            line = i
            file.write(f"{i}\n")
            print(f"\nResult: {line}")
            file.flush()

            # Ждем одну секунду
            time.sleep(1)

    # Пример работы с моделями семейства Qwen2-VL
    cache_directory = "model_cache"

    # Сохраняем модели Qwen2-VL в примонтированную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Имена моделей и семейство моделей
    model_name_1 = "Qwen2-VL-2B-Instruct"
    model_family = "Qwen2-VL"

    # Инфо о том где взять класс для семейства моделей
    package = "model_qwen2_vl"
    module = "models"
    model_class = "Qwen2VLModel"
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(model_family, model_class_path)

    image_path = "example_docs/schet_na_oplatu.png"
    question = "Пожалуйста собери следующую информацию с документа:покупатель,\r\nИНН покупателя,\r\nКПП покупателя,\r\nтелефон покупателя,\r\nпоставщик,\r\nИНН поставщика,\r\nБИК поставщика,\r\nКор. счет поставщика,\r\nР/с поставщика,\r\nдата документа,\r\nномер счета документа,\r\nПеречисли каждый купленный товар  (наименование, количество, цена за штуку)\r\nкакую сумму нужно заплатить за все товары\r\nв какой валюте платим,\r\nвзнос НДС.\r\nВерни ответ в виде json файла с полями и ответами на них."

    # создаем модель
    model_init_params = {
        "model_name": model_name_1,
        "system_prompt": "",
        "cache_dir": "model_cache",
    }

    model = ModelFactory.get_model(model_family, model_init_params)

    # отвечаем на вопрос о по одной картинке
    model_answer = model.predict_on_image(image=image_path, question=question)

    # отвечаем на вопрос о по нескольким картинкам сразу (пока не реализован)
    # model_answer = model.predict_on_images(images=[image_path1, image_path2], question=question)

    print(model_answer)
