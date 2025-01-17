import os

from benchmark_scheduler.benchmark_scheduler import run_container


if __name__ == "__main__":
    # Путь к папке на хосте и путь внутри контейнера
    host_directory = os.path.join(os.getcwd(), "pipeline")
    container_directory = "/workspace"

    vlm_docker_img = "ghcr.io/vlmhyperbenchteam/qwen2-vl:ubuntu22.04-cu124-torch2.4.0_v0.1.0"
    eval_docker_img = "ghcr.io/vlmhyperbenchteam/qwen2-vl:ubuntu22.04-cu124-torch2.4.0_v0.1.0"

    run_container(
        vlm_docker_img,
        host_dir=host_directory,
        container_dir=container_directory,
        script_path="/workspace/scripts/run_vlm.py",
        packages_to_install=["wheels/some_package-0.1.0-py3-none-any.whl"],
        keep_container=False,
        use_gpu=True,
    )

    run_container(
        vlm_docker_img,
        host_dir=host_directory,
        container_dir=container_directory,
        script_path="/workspace/scripts/run_eval.py",
        packages_to_install=["wheels/some_package-0.1.0-py3-none-any.whl"],
        keep_container=False,
    )
