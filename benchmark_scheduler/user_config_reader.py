import ast
from typing import List

import numpy as np
import pandas as pd
from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig


class UserConfigReader:
    """
    Класс для считывания параметров `Evaluation run` указанных в `user_config.csv`.
    Дополняет конфигурацию "пробега модели по датасету" параметрами из Базы VLM.

    Attributes:
        cfg_df (pd.DataFrame): DataFrame с данными из пользовательской конфигурации.
        vlm_df (pd.DataFrame): DataFrame с параметрами из Базы VLM.
    """

    def __init__(
        self, config_path: str, vlm_base_path: str = "vlmhyperbench/vlm_base.csv"
    ) -> None:
        """
        Инициализирует объект UserConfigReader.

        Args:
            config_path (str): Путь к файлу пользовательской конфигурации (user_config.csv).
            vlm_base_path (str): Путь к файлу Базы VLM. Defaults to "vlmhyperbench/vlm_base.csv".
        """
        self.cfg_df = self.load_user_config(config_path)
        self.vlm_df = self.load_vlm_base(vlm_base_path)

    @staticmethod
    def load_user_config(config_path: str) -> pd.DataFrame:
        """
        Загружает пользовательскую конфигурацию (user_config.csv) из CSV-файла.

        Args:
            config_path (str): Путь к файлу пользовательской конфигурации.

        Returns:
            pd.DataFrame: DataFrame с данными из пользовательской конфигурации.
        """
        try:
            df_cfg = pd.read_csv(config_path, sep=";", encoding="utf-8-sig")
            df_cfg["metrics"] = df_cfg["metrics"].apply(ast.literal_eval)
            df_cfg["metrics_aggregators"] = df_cfg["metrics_aggregators"].apply(
                ast.literal_eval
            )
            df_cfg.replace({np.nan: None}, inplace=True)
            return df_cfg

        except Exception as e:
            print(f"Error loading user config: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_vlm_base(vlm_base_path: str) -> pd.DataFrame:
        """
        Загружает параметры VLM из Базы VLM в виде csv-файла.

        Args:
            vlm_base_path (str): Путь к файлу Базы VLM.

        Returns:
            pd.DataFrame: DataFrame с параметрами из Базы VLM.
        """
        try:
            df_vlm_base = pd.read_csv(vlm_base_path, sep=";", encoding="utf-8-sig")
            df_vlm_base.replace({np.nan: None}, inplace=True)
            return df_vlm_base

        except Exception as e:
            print(f"Error loading VLM base: {e}")
            return pd.DataFrame()

    def read_user_config(self) -> List[BenchmarkRunConfig]:
        """
        Читает пользовательскую конфигурацию и создаёт список конфигураций для запуска бенчмарков.

        Returns:
            List[BenchmarkRunConfig]: Список из BenchmarkRunConfig.
        """
        bench_run_cfgs = []

        for row in self.cfg_df.itertuples():
            row_dict = row._asdict()

            # Чтение параметров из vlm_base.csv
            vlm_params = self.vlm_df[
                (self.vlm_df["framework"] == row_dict["framework"])
                & (self.vlm_df["model_name"] == row_dict["model_name"])
            ].to_dict(orient="records")[0]

            bench_run_cfg = BenchmarkRunConfig(
                # Устанавливаем параметры из user_config.csv
                dataset=row_dict["dataset"],
                framework=row_dict["framework"],
                model_name=row_dict["model_name"],
                task_name=row_dict["task"],
                system_prompt=row_dict["system_prompt"],
                prompt_collection=row_dict["prompt_collection"],
                metrics=row_dict["metrics"],
                only_evaluate_metrics=row_dict["only_evaluate_metrics"],
                metrics_aggregators=row_dict["metrics_aggregators"],
                filter_doc_class=row_dict["filter_doc_class"],
                filter_question_type=row_dict["filter_question_type"],
                # Устанавливаем параметры из vlm_base.csv
                model_family=vlm_params["model_family"],
                docker_image=vlm_params["docker_image"],
                git_python_package=vlm_params["git_python_package"],
                python_package=vlm_params["python_package"],
                module=vlm_params["module"],
                class_name=vlm_params["class_name"],
            )

            bench_run_cfgs.append(bench_run_cfg)

        return bench_run_cfgs
