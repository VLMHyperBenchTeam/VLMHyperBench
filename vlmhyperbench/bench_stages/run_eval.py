import os

from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from metric_evaluator.metric_evaluator import MetricEvaluator
from config_manager.config_manager import ConfigManager


def run_vlm_eval_greet(config, model_class_path):
    print("\n" * 2)
    print("#" * 80)
    print("Оценка модели:")
    print(f"dataset: {config.dataset}\n")
    print(f"framework: {config.framework}")
    print(f"model_name: {config.model_name}")
    print(f"model_family: {config.model_family}\n")
    print(f"docker_image: {config.docker_image}")
    print(f"model_class_path: {model_class_path}\n")
    print(f"task_name: {config.task_name}")
    print(f"system_prompt: {config.system_prompt}")
    print(f"prompt_collection: {config.prompt_collection}")
    print(f"metrics: {config.metrics}")
    print(f"metrics_aggregators: {config.metrics_aggregators}")
    print(f"filter_doc_class: {config.filter_doc_class}")
    print(f"filter_question_type: {config.filter_question_type}")
    print("#" * 80)
    print("\n" * 2)


if __name__ == "__main__":
    # TODO: получился жестко захардкоженный параметр!
    cfg_path = "/workspace/cfg/VLMHyperBench_config.json"
    run_cfg_filename = "BenchmarkRunConfig.json"
    
    # Получаем конфиг с всеми путями в Docker-контейнере
    software_cfg = ConfigManager(cfg_path)
    container_cfg = software_cfg.cfg_container
    
    # Получаем конфиг со всеми параметрами прогона
    config_dir = os.path.join(container_cfg["system_dirs"]["cfg"], run_cfg_filename)
    print(config_dir)
    config = BenchmarkRunConfig.from_json(config_dir)

    # Инфо о том где взять класс для семейства моделей
    model_class_path = f"{config.python_package}.{config.module}:{config.class_name}"

    run_vlm_eval_greet(config, model_class_path)

    dataset_annot = os.path.join(
        container_cfg["data_dirs"]["datasets"], config.task_name, config.dataset, "annotation.csv"
    )

    # Получаем название файла метрик
    model_answers = config.metric_file
    filename_from_metric = os.path.split(model_answers)[-1]
    filename_from_metric = os.path.splitext(filename_from_metric)[0]

    metric_eval = MetricEvaluator(dataset_annot, model_answers)

    # Производим расчет метрик по всем агрегаторам
    for metrics_aggregator in config.metrics_aggregators:
        metric_csv_path = os.path.join(
            container_cfg["data_dirs"]["model_metrics"], f"{filename_from_metric}_{metrics_aggregator}.csv"
        )
        metric_eval.save_function_results(
            csv_path=metric_csv_path,
            func_name=metrics_aggregator,
            metrics=config.metrics,
        )
