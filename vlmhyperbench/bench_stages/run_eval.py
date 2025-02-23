import os

from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from metric_evaluator.metric_evaluator import MetricEvaluator

if __name__ == "__main__":

    run_cfg_dir = "cfg"  # папка с BenchmarkRunConfig.json
    cache_directory = "model_cache"  # сохраняем VLM в эту папку
    run_cfg_filename = "BenchmarkRunConfig.json"

    # Получаем конфиг со всеми параметрами прогона
    config_dir = os.path.join(run_cfg_dir, run_cfg_filename)
    config = BenchmarkRunConfig.from_json(config_dir)

    dataset_annot = os.path.join("Datasets", config.task_name, config.dataset, "annotation.csv")
    model_answers = config.metric_file
    date_from_metric = os.path.split(model_answers)[-1]
    date_from_metric = os.path.splitext(date_from_metric)[0]
    date_from_metric = date_from_metric.split("_")[-2:]
    date_from_metric = "_".join(date_from_metric)

    metric_eval = MetricEvaluator(dataset_annot, model_answers)

    df_by_id_path_csv = f"ModelsMetrics/by_id_{date_from_metric}.csv"
    df_by_id = metric_eval.save_function_results(
        csv_path=df_by_id_path_csv, func_name="by_id"
    )

    df_by_doc_type_path_csv = f"ModelsMetrics/df_by_doc_type_{date_from_metric}.csv"
    metric_eval.save_function_results(
        csv_path=df_by_doc_type_path_csv, func_name="by_doc_type", func_arg=df_by_id
    )

    df_by_doc_question_path_csv = f"ModelsMetrics/df_by_doc_question_{date_from_metric}.csv"
    metric_eval.save_function_results(
        csv_path=df_by_doc_question_path_csv,
        func_name="by_doc_question",
        func_arg=df_by_id,
    )

    df_general_csv_path = f"ModelsMetrics/df_general_{date_from_metric}.csv"
    df_general = metric_eval.save_function_results(
        csv_path=df_general_csv_path, func_name="general"
    )
