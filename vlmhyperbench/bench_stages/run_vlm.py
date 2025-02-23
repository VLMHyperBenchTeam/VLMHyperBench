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

    # параметры модели - передаем системный промпт, если доступен
    model_init_params = {
        "model_name": config.model_name,
        "system_prompt": "",
        "cache_dir": cache_directory,
    }

    run_vlm_stage_greet(config, model_class_path)

    model = ModelFactory.get_model(config.model_family, model_init_params)

    # Совершаем прогон модели по датасету
    dataset_dir_path = os.path.join("Datasets", config.task_name, config.dataset)
    iterator = IteratorFabric.get_dataset_iterator(
        task_name=config.task_name,
        dataset_name=config.dataset,
        filter_doc_class=config.filter_doc_class,
        filter_question_type=config.filter_question_type,
        dataset_dir_path=dataset_dir_path,
        prompt_collection_filename=config.prompt_collection,
        prompt_dir="PromptCollection",
    )
    runner = IteratorFabric.get_runner(
        iterator, model, answers_dir_path="ModelsAnswers"
    )
    runner.run()
    metric_file_path = runner.save_answers()
    
    config.metric_file = metric_file_path
    
    config.to_json(config_dir)
