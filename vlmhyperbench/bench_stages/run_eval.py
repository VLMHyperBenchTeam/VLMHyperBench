import os

from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from metric_evaluator.metric_evaluator import MetricEvaluator


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

    run_cfg_dir = "cfg"  # папка с BenchmarkRunConfig.json
    run_cfg_filename = "BenchmarkRunConfig.json"
    model_metrics_dir = "ModelsMetrics"

    # Получаем конфиг со всеми параметрами прогона
    config_dir = os.path.join(run_cfg_dir, run_cfg_filename)
    config = BenchmarkRunConfig.from_json(config_dir)
    
    # Инфо о том где взять класс для семейства моделей
    model_class_path = f"{config.python_package}.{config.module}:{config.class_name}"
    
    run_vlm_eval_greet(config, model_class_path)

    dataset_annot = os.path.join(
        "Datasets", config.task_name, config.dataset, "annotation.csv"
    )
    
    model_answers = config.metric_file
    filename_from_metric = os.path.split(model_answers)[-1]
    filename_from_metric = os.path.splitext(filename_from_metric)[0]

    metric_eval = MetricEvaluator(dataset_annot, model_answers)

    df_by_id_csv_path = os.path.join(model_metrics_dir, f"{filename_from_metric}_by_id.csv")
    df_by_id = metric_eval.save_function_results(
        csv_path=df_by_id_csv_path, func_name="by_id", metrics=config.metrics
    )

    df_by_doc_type_csv_path = os.path.join(model_metrics_dir, f"{filename_from_metric}_df_by_doc_type.csv")
    metric_eval.save_function_results(
        csv_path=df_by_doc_type_csv_path,
        func_name="by_doc_type",
        func_arg=df_by_id,
        metrics=config.metrics,
    )

    df_by_doc_question_csv_path = os.path.join(model_metrics_dir, f"{filename_from_metric}_df_by_doc_question.csv")
    metric_eval.save_function_results(
        csv_path=df_by_doc_question_csv_path,
        func_name="by_doc_question",
        func_arg=df_by_id,
        metrics=config.metrics,
    )

    df_general_csv_path = os.path.join(model_metrics_dir, f"{filename_from_metric}_general.csv")
    df_general = metric_eval.save_function_results(
        csv_path=df_general_csv_path, func_name="general", metrics=config.metrics
    )
