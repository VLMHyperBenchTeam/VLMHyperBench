from benchmark_scheduler.benchmark_orchestrator import BenchmarkOrchestrator

if __name__ == "__main__":
    # Загрузка конфига из файла
    cfg_path = "vlmhyperbench/system_dirs/cfg/VLMHyperBench_config.json"

    eval_orchestrator = BenchmarkOrchestrator(cfg_path)
    eval_orchestrator.run_scheduler()