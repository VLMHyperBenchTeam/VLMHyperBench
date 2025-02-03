import os

from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from dataset_iterator.fabrics import IteratorFabric
from model_interface.model_factory import ModelFactory


def run_vlm_stage_greet(config, model_class_path):
    print("\n" * 2)
    print("#" * 80)
    print("Запуск модели:")
    print(f"dataset: {config.dataset}\n")
    print(f"framework: {config.framework}")
    print(f"model_name: {config.model_name}")
    print(f"model_family: {config.model_family}\n")
    print(f"docker_image: {config.docker_image}")
    print(f"model_class_path: {model_class_path}\n")
    print(f"task_name: {config.task_name}")
    print(f"prompt_collection: {config.prompt_collection}")
    print("#" * 80)
    print("\n" * 2)


_system_prompt_cls_example = """
Задача: Определите тип каждого документа на предоставленных изображениях и выведите их в виде последовательности цифр, где каждая цифра соответствует определенному типу документа. Ответ должен содержать только порядок цифр, без дополнительного текста.
Типы документов:
1 - old_tins: инн старого образца
2 - new_tins: инн нового образца
3 - interest_free_loan_agreement: договор о беспроцентном займе
4 - snils: снилс
5 - invoice: счет фактура
6 - passport: паспорт России
Пример ответа: 2,4,5,1,3
Пожалуйста, предоставьте ответ в указанном формате.
"""

_system_prompt_srt_example = """Расставь страницы в правильном порядке, используя только цифры, разделенные запятыми. Например: 2,4,1,3,5."""

if __name__ == "__main__":
    run_cfg_dir = "cfg"  # папка с BenchmarkRunConfig.json
    cache_directory = "model_cache"  # сохраняем VLM в эту папку
    run_cfg_filename = "BenchmarkRunConfig.json"

    # Получаем конфиг со всеми параметрами прогона
    config_dir = os.path.join(run_cfg_dir, run_cfg_filename)
    config = BenchmarkRunConfig.from_json(config_dir)

    # Инфо о том где взять класс для семейства моделей
    model_class_path = f"{config.python_package}.{config.module}:{config.class_name}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(config.model_family, model_class_path)

    # TODO: Получаем системный промпт
    # system_prompt_adapter = ...

    ##### CLASSIFICATION SECTION
    task_name = "RPOClassification"
    # параметры модели - передаем системный промпт, если доступен
    model_init_params = {
        "model_name": config.model_name,
        "system_prompt": _system_prompt_cls_example,
        "cache_dir": cache_directory,
    }

    run_vlm_stage_greet(config, model_class_path)

    model = ModelFactory.get_model(config.model_family, model_init_params)

    # Совершаем прогон модели по датасету
    cls_iterator = IteratorFabric.get_dataset_iterator(
                                                    dataset_dir_path=r"/workspace/Datasets/rpo",
                                                    task_name=task_name,
                                                    dataset_name=config.dataset,
                                                    filter_doc_class=config.filter_doc_class,
                                                    filter_question_type=config.filter_question_type,
                                                    prompt_file_dir=r"prompts",
                                                    prompt_file_name=r"my_prompt.txt",
                                                    )


    cls_runner = IteratorFabric.get_runner(
        cls_iterator, model, answers_dir_path="ModelsAnswers"
    )
    cls_runner.run()
    save_path_classification = cls_runner.save_answers()
    print("Ответы сохранены в", save_path_classification)


    #####  SORTING SECTION
    task_name = "RPOClassification"

    # создаем новый итератор
    sort_iterator = IteratorFabric.get_dataset_iterator(task_name=task_name,
                                                       dataset_name=config.dataset,
                                                       dataset_dir_path=r"/workspace/Datasets/rpo",
                                                       prompt_file_dir=r"prompts",
                                                       prompt_file_name=r"my_prompt.txt",
                                                       )
     

    # Получаем раннер
    sort_runner = IteratorFabric.get_runner(iterator=sort_iterator, 
                                           model=model,
                                           answers_dir_path="ModelsAnswers",
                                           classification_answers_path=save_path_classification)

    # Совершаем прогон по датасету
    sort_runner.run()
    # Сохраняем ответы
    save_path = sort_runner.save_answers()
    print("Ответы сохранены в", save_path)
