import os

from benchmark_scheduler.benchmark_scheduler import run_container


if __name__ == "__main__":
    # Путь к папке на хосте и путь внутри контейнера
    host_directory = os.path.join(
        os.getcwd(), "pipeline"
    )  # Замените на реальный путь на хосте
    container_directory = "/workspace"  # Замените на желаемый путь в контейнере

    image_name = "vlmevalkit:v0.2rc1-cu124"

    run_container(
        image_name,
        host_dir=host_directory,
        container_dir=container_directory,
        script_path="/workspace/scripts/writer.py",
        packages_to_install=["wheels/some_package-0.1.0-py3-none-any.whl"],
    )

    run_container(
        image_name,
        host_dir=host_directory,
        container_dir=container_directory,
        script_path="/workspace/scripts/reader.py",
        use_gpu=True,
        packages_to_install=["wheels/some_package-0.1.0-py3-none-any.whl"],
    )
