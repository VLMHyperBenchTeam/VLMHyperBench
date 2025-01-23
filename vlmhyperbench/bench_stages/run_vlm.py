import os

from model_interface.model_factory import ModelFactory
from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from dataset_iterator.fabrics import IteratorFabric

if __name__ == "__main__":

    # Пример работы с моделями семейства Qwen2-VL
    cache_directory = "model_cache"

    # Сохраняем модели Qwen2-VL в примонтированную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Получаем конфиг со всеми параметрами прогона
    config_dir = os.path.join(script_dir, "BenchmarkRunConfig.json")
    config = BenchmarkRunConfig.from_json(config_dir)

    # Инфо о том где взять класс для семейства моделей
    model_class_path = f"{config.python_package}.{config.module}:{config.class_name}"

    # Регистрация модели в фабрике
    model_family = "Qwen2-VL"
    ModelFactory.register_model(model_family, model_class_path)

    # TODO: Получаем системный промпт
    # system_prompt_adapter = ...

    # параметры модели - передаем системный промпт, если доступен
    model_init_params = {
        "model_name": config.model_name,
        "system_prompt": "",
        "cache_dir": "model_cache",
    }

    model = ModelFactory.get_model(model_family, model_init_params)

    # Совершаем прогон по датасету
    task_name = "VQA"
    iterator = IteratorFabric.get_dataset_iterator(task_name=task_name, 
                                                   dataset_name=config.dataset,
                                                   filter_doc_class=config.filter_doc_class,
                                                   filter_question_type=config.filter_question_type)
    runner = IteratorFabric.get_runner(iterator, model)
    runner.run()
    # отвечаем на вопрос о по одной картинке
    # image_path = "example_docs/schet_na_oplatu.png"
    # question = "Пожалуйста собери следующую информацию с документа:покупатель,\r\nИНН покупателя,\r\nКПП покупателя,\r\nтелефон покупателя,\r\nпоставщик,\r\nИНН поставщика,\r\nБИК поставщика,\r\nКор. счет поставщика,\r\nР/с поставщика,\r\nдата документа,\r\nномер счета документа,\r\nПеречисли каждый купленный товар  (наименование, количество, цена за штуку)\r\nкакую сумму нужно заплатить за все товары\r\nв какой валюте платим,\r\nвзнос НДС.\r\nВерни ответ в виде json файла с полями и ответами на них."

    # model_answer = model.predict_on_image(image=image_path, question=question)

    # отвечаем на вопрос о по нескольким картинкам сразу (пока не реализован)
    # model_answer = model.predict_on_images(images=[image_path1, image_path2], question=question)

    # print(model_answer)
