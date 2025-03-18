from benchmark_scheduler.benchmark_orchestrator import (
    host_paths_to_abs,
    load_env_vars,
    run_container,
)
from config_manager.config_manager import ConfigManager


if __name__ == "__main__":
    # Загрузка конфига из файла
    cfg_path = "vlmhyperbench/cfg/VLMHyperBench_config.json"

    # Считываем конфиг для VLMHyperBench
    config = ConfigManager(cfg_path)

    # Получим маппинг директорий для Docker-контейнера
    volumes = config.get_volumes()
    volumes = host_paths_to_abs(volumes)
    
    # Получим список python-пакетов для каждого этапа работы
    vlm_run_packages = config.load_packages("vlm_run")
    eval_run_packages = config.load_packages("eval_run")

    environment = load_env_vars()

    # 2. Этап "Запуск VLM" на Docker-контейнере
    config.vlm_run_packages.append()
    run_container(
        vlm_docker_img,
        volumes,
        script_path="/workspace/bench_stages/run_vlm.py",
        packages_to_install=config.vlm_run_packages,
        use_gpu=True,
    )

    # 3. Этап "Оценка метрик" на Docker-контейнере
    run_container(
        config.cfg["eval_docker_img"],
        volumes,
        script_path="bench_stages/run_eval.py",
        packages_to_install=eval_run_packages,
    )
