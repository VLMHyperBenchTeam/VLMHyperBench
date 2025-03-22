import os
from config_manager.config_manager import ConfigManager

from benchmark_scheduler.benchmark_orchestrator import (
    host_paths_to_abs,
    load_env_vars,
    run_container,
)
from benchmark_scheduler.user_config_reader import UserConfigReader

if __name__ == "__main__":
    # Загрузка конфига из файла
    cfg_path = "vlmhyperbench/system_dirs/cfg/VLMHyperBench_config.json"

    # Считываем конфиг для VLMHyperBench
    evalkit_config = ConfigManager(cfg_path)

    # Получим маппинг директорий для Docker-контейнера
    volumes = evalkit_config.get_volumes()
    volumes = host_paths_to_abs(volumes)

    # Получим список python-пакетов для каждого этапа работы
    vlm_run_packages = evalkit_config.load_packages("vlm_run")
    eval_run_packages = evalkit_config.load_packages("eval_run")

    # загружаем переменные окружения
    environment = load_env_vars()
    environment["VLMHYPERBENCH_CONFIG_PATH"] = evalkit_config.cfg_container["vlmhyperbench_cfg"]

    # Получим список Evaluation Run'ов из `user_config.csv`
    user_cfg_reader = UserConfigReader(
        evalkit_config.cfg["user_config"], evalkit_config.cfg["vlm_base"]
    )
    bench_run_cfgs = user_cfg_reader.read_user_config()

    for bench_run_cfg in bench_run_cfgs:
        bench_run_cfg.to_json(evalkit_config.cfg["benchmark_run_cfg"])

        # 2. Этап "Запуск VLM" на Docker-контейнере
        vlm_run_packages.append(bench_run_cfg.git_python_package)
        print(vlm_run_packages)
        run_vlm_path = os.path.join(
            evalkit_config.cfg_container["system_dirs"]["bench_stages"], "run_vlm.py"
        )
        run_container(
            bench_run_cfg.docker_image,
            volumes,
            script_path=run_vlm_path,
            packages_to_install=vlm_run_packages,
            use_gpu=True,
            environment=environment
        )

        # 3. Этап "Оценка метрик" на Docker-контейнере
        print(evalkit_config.cfg["eval_docker_img"])
        run_eval_path = os.path.join(
            evalkit_config.cfg_container["system_dirs"]["bench_stages"], "run_eval.py"
        )
        run_container(
            evalkit_config.cfg["eval_docker_img"],
            volumes,
            script_path=run_eval_path,
            packages_to_install=eval_run_packages,
            environment=environment
        )
