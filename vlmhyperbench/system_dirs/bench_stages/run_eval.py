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
    # Получаем конфиг с всеми путями в Docker-контейнере
    cfg_path = os.getenv("VLMHYPERBENCH_CONFIG_PATH")
    cfg_container = ConfigManager(cfg_path).cfg_container
    
    # Получаем конфиг со всеми параметрами прогона
    run_config = BenchmarkRunConfig.from_json(cfg_container["benchmark_run_cfg"])

    # Инфо о том где взять класс для семейства моделей
    model_class_path = f"{run_config.python_package}.{run_config.module}:{run_config.class_name}"

    run_vlm_eval_greet(run_config, model_class_path)

    dataset_annot = os.path.join(
        cfg_container["data_dirs"]["datasets"], run_config.task_name, run_config.dataset, "annotation.csv"
    )

    # Получаем название файла метрик
    model_answers = run_config.metric_file
    filename_from_metric = os.path.split(model_answers)[-1]
    filename_from_metric = os.path.splitext(filename_from_metric)[0]

    metric_eval = MetricEvaluator(dataset_annot, model_answers)

    # Производим расчет метрик по всем агрегаторам
    for metrics_aggregator in run_config.metrics_aggregators:
        metric_csv_path = os.path.join(
            cfg_container["data_dirs"]["model_metrics"], f"{filename_from_metric}_{metrics_aggregator}.csv"
        )
        metric_eval.save_function_results(
            csv_path=metric_csv_path,
            func_name=metrics_aggregator,
            metrics=run_config.metrics,
        )
