import os

from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from config_manager.config_manager import ConfigManager
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
    print(f"system_prompt: {config.system_prompt}")
    print(f"prompt_collection: {config.prompt_collection}")
    print("#" * 80)
    print("\n" * 2)


if __name__ == "__main__":
    # Получаем конфиг с всеми путями в Docker-контейнере
    cfg_path = os.getenv("VLMHYPERBENCH_CONFIG_PATH")
    cfg_container = ConfigManager(cfg_path).cfg_container

    # Получаем конфиг со всеми параметрами прогона
    run_config = BenchmarkRunConfig.from_json(cfg_container["benchmark_run_cfg"])

    # Инфо о том где взять класс для семейства моделей
    model_class_path = (
        f"{run_config.python_package}.{run_config.module}:{run_config.class_name}"
    )

    # Регистрация модели в фабрике
    ModelFactory.register_model(run_config.model_family, model_class_path)

    # TODO: Получаем системный промпт
    # system_prompt_adapter = ...

    # параметры модели - передаем системный промпт, если доступен
    model_init_params = {
        "model_name": run_config.model_name,
        "system_prompt": "",
        "cache_dir": cfg_container["system_dirs"]["model_cache"],
    }

    run_vlm_stage_greet(run_config, model_class_path)

    model = ModelFactory.get_model(run_config.model_family, model_init_params)

    # Совершаем прогон модели по датасету
    dataset_dir_path = os.path.join(
        cfg_container["data_dirs"]["datasets"], run_config.task_name, run_config.dataset
    )

    prompt_collection_dir_path = os.path.join(
        cfg_container["data_dirs"]["prompt_collections"], run_config.task_name
    )
    print(dataset_dir_path)
    print(prompt_collection_dir_path)

    iterator = IteratorFabric.get_dataset_iterator(
        task_name=run_config.task_name,
        dataset_name=run_config.dataset,
        filter_doc_class=run_config.filter_doc_class,
        filter_question_type=run_config.filter_question_type,
        dataset_dir_path=dataset_dir_path,
        prompt_collection_filename=run_config.prompt_collection,
        prompt_dir=prompt_collection_dir_path,
    )

    runner = IteratorFabric.get_runner(
        iterator, model, answers_dir_path=cfg_container["data_dirs"]["model_answers"]
    )
    runner.run()

    metric_file_path = runner.save_answers()

    # Сохраняем результаты в BenchmarkRunConfig.json
    run_config.metric_file = metric_file_path
    run_config.to_json(cfg_container["benchmark_run_cfg"])
