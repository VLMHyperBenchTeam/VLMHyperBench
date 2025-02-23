import os

from benchmark_scheduler.benchmark_orchestrator import (
    host_paths_to_abs,
    load_env_vars,
    run_container,
)


if __name__ == "__main__":
    vlm_docker_img = (
        "ghcr.io/vlmhyperbenchteam/qwen2.5-vl:ubuntu22.04-cu124-torch2.4.0_v0.1.0"
    )
    eval_docker_img = (
        "ghcr.io/vlmhyperbenchteam/metric-evaluator:python3.10-slim_v0.1.0"
    )

    environment = load_env_vars()

    volumes = {
        # папки для обмена файлами
        "vlmhyperbench/cfg": "/workspace/cfg",
        "vlmhyperbench/SystemPrompts": "/workspace/SystemPrompts",
        "vlmhyperbench/PromptCollection": "/workspace/PromptCollection",
        "vlmhyperbench/Datasets": "/workspace/Datasets",
        "vlmhyperbench/ModelsAnswers": "/workspace/ModelsAnswers",
        "vlmhyperbench/ModelsMetrics": "/workspace/ModelsMetrics",
        # служебные папки EvalKit
        "vlmhyperbench/bench_stages": "/workspace/bench_stages",
        "vlmhyperbench/wheels": "/workspace/wheels",
        "vlmhyperbench/model_cache": "/workspace/model_cache",
    }

    volumes = host_paths_to_abs(volumes, current_dir=None)

    run_container(
        vlm_docker_img,
        volumes,
        script_path="bench_stages/run_vlm.py",
        packages_to_install=[
            "git+https://github.com/VLMHyperBenchTeam/benchmark_run_config.git@0.1.3",
            # "/workspace/wheels/benchmark_run_config-0.1.0-py3-none-any.whl",
            "git+https://github.com/VLMHyperBenchTeam/model_interface.git@0.1.0",
            "git+https://github.com/VLMHyperBenchTeam/model_qwen2.5-vl.git@0.1.0",
            "git+https://github.com/VLMHyperBenchTeam/dataset_iterator.git@0.2.1",
            # "git+https://github.com/VLMHyperBenchTeam/system_prompt_adapter.git@0.1.0",
        ],
        use_gpu=True,
    )

    run_container(
        eval_docker_img,
        volumes,
        script_path="bench_stages/run_eval.py",
        packages_to_install=[
            "git+https://github.com/VLMHyperBenchTeam/benchmark_run_config.git@0.1.3",
            "git+https://github.com/VLMHyperBenchTeam/metric_evaluator.git@0.1.0",
        ],
    )
