import copy
import os
from typing import Any, Dict, List, Optional

import docker
from config_manager.config_manager import ConfigManager
from docker.errors import APIError, ImageNotFound
from dotenv import load_dotenv
from tqdm import tqdm

from .user_config_reader import UserConfigReader


class BenchmarkOrchestrator:
    """Оркестратор для выполнения `Evaluation run` для VLM-моделей.

    Attributes:
        evalkit_config (ConfigManager): Менеджер конфигурации VLMHyperBench
        volumes (Dict[str, str]): Словарь с маппингом директорий для монтирования в Docker-контейнеры
        vlm_run_packages (List[str]): Список Python-пакетов, которые будут установлены для этапа запуска VLM
        eval_run_packages (List[str]): Список Python-пакетов, которые будут установлены для этапа оценки метрик
        environment (Dict[str, str]): Переменные окружения для контейнеров
        bench_run_cfgs (List[BenchmarkRunConfig]): Список конфигураций запусков бенчмарка

    """

    def __init__(self, config_path: str) -> None:
        """Инициализирует оркестратор.

        Args:
            config_path (str): Путь к конфигурационному файлу VLMHyperBench

        """
        # Считываем конфиг для VLMHyperBench
        self.evalkit_config = ConfigManager(config_path)

        # Получим маппинг директорий для Docker-контейнера
        self.volumes = self.evalkit_config.get_volumes()
        self.volumes = host_paths_to_abs(self.volumes)

        # Получим список python-пакетов для каждого этапа работы
        self.vlm_run_packages = self.evalkit_config.load_packages("vlm_run")
        self.eval_run_packages = self.evalkit_config.load_packages("eval_run")

        # загружаем переменные окружения
        self.environment = load_env_vars()
        self.environment["VLMHYPERBENCH_CONFIG_PATH"] = (
            self.evalkit_config.cfg_container["vlmhyperbench_cfg"]
        )

        # Получим список Evaluation Run'ов из `user_config.csv`
        user_cfg_reader = UserConfigReader(
            self.evalkit_config.cfg["user_config"], self.evalkit_config.cfg["vlm_base"]
        )
        self.bench_run_cfgs = user_cfg_reader.read_user_config()

    def run_scheduler(self):
        """Запускает цикл выполнения всех конфигураций `Evaluation run`.

        Последовательно выполняет benchmark_run() для каждой конфигурации
        из self.bench_run_cfgs.

        Перед запуском `Evaluation run` проверяет наличие Docker-образов.
        """
        if not self.pull_required_images():
            print("Ошибка: Не все необходимые Docker-образы были загружены.")
            return
        for bench_run_cfg in self.bench_run_cfgs:
            self.benchmark_run(bench_run_cfg)

    def benchmark_run(self, bench_run_cfg):
        """Выполняет полный цикл одного `Evaluation run`.

        Args:
            bench_run_cfg (BenchmarkRunConfig): Конфигурация текущего запуска

        Осуществляет:
            1. Сохранение конфигурации в JSON
            2. Запуск VLM-модели в Docker-контейнере
            3. Оценку метрик в отдельном Docker-контейнере

        """
        bench_run_cfg.to_json(self.evalkit_config.cfg["benchmark_run_cfg"])

        # 2. Этап "Запуск VLM" на Docker-контейнере
        vlm_run_packages = copy.deepcopy(self.vlm_run_packages)
        vlm_run_packages.append(bench_run_cfg.git_python_package)
        print(vlm_run_packages)
        run_vlm_path = os.path.join(
            self.evalkit_config.cfg_container["system_dirs"]["bench_stages"],
            "run_vlm.py",
        )
        run_container(
            bench_run_cfg.docker_image,
            self.volumes,
            script_path=run_vlm_path,
            packages_to_install=vlm_run_packages,
            use_gpu=True,
            environment=self.environment,
        )

        # 3. Этап "Оценка метрик" на Docker-контейнере
        print(self.evalkit_config.cfg["eval_docker_img"])
        run_eval_path = os.path.join(
            self.evalkit_config.cfg_container["system_dirs"]["bench_stages"],
            "run_eval.py",
        )
        run_container(
            self.evalkit_config.cfg["eval_docker_img"],
            self.volumes,
            script_path=run_eval_path,
            packages_to_install=self.eval_run_packages,
            environment=self.environment,
        )

    def pull_required_images(self) -> bool:
        """Скачивает все необходимые Docker-образы для выполнения `Evaluation run`.

        Returns:
            bool: True, если все образы успешно загружены, иначе False.
        """
        required_images = set()
        # Собираем образы из конфигураций Evaluation run
        for bench_run_cfg in self.bench_run_cfgs:
            required_images.add(bench_run_cfg.docker_image)

        # Добавляем образ для оценки метрик
        required_images.add(self.evalkit_config.cfg["eval_docker_img"])

        success = True
        for image in required_images:
            # Разделяем имя образа и тег
            parts = image.split(":", 1)
            image_name = parts[0]
            tag = parts[1] if len(parts) > 1 else "latest"

            print(f"Проверка образа: {image_name}:{tag}")
            if not check_and_pull_image(image_name, tag):
                success = False
        return success


def host_paths_to_abs(
    volumes: Dict[str, str], current_dir: str | None = None
) -> Dict[str, str]:
    """Делает пути на хосте абсолютными в словаре volumes.
    Это необходимо для монтирования данных каталогов к запускаемому Docker-контейнеру.

    Если current_dir не указан, используется текущая рабочая директория.

    Args:
        volumes (Dict[str, str]): Словарь, осуществляет mapping директорий, которые будут примонтированы
            к запущенному Docker-контейнеру.
            В этом словаре: ключи — пути на хосте, а значения — пути внутри контейнера.
        current_dir (str | None, optional): Текущая директория. Если None, используется os.getcwd().
            По умолчанию None.

    Returns:
        Dict[str, str]: Новый словарь, где ключи — абсолютные пути на хосте, а значения — пути внутри контейнера.

    Example:
        volumes = {
            "pipeline/data": "/workspace/data",
            "pipeline/bench_stages": "/workspace/bench_stages",
            "pipeline/wheels": "/workspace/wheels",
        }
        current_dir = "/home/user/project"
        result = add_current_dir_to_volumes(volumes, current_dir)
        print(result)
        # {
        #     "/home/user/project/pipeline/data": "/workspace/data",
        #     "/home/user/project/pipeline/bench_stages": "/workspace/bench_stages",
        #     "/home/user/project/pipeline/wheels": "/workspace/wheels",
        # }
    """
    if current_dir is None:
        current_dir = os.getcwd()

    return {
        os.path.join(current_dir, host_path): container_path
        for host_path, container_path in volumes.items()
    }


def check_and_pull_image(image_name: str, tag: str = "latest") -> bool:
    """Проверяет наличие Docker-образа и скачивает его при необходимости.

    Args:
        image_name (str): Название Docker-образа (например, "python")
        tag (str): Тег образа. По умолчанию "latest"

    Returns:
        bool: True если образ успешно найден/скачан, False в случае ошибки

    Raises:
        docker.errors.APIError: При ошибках API Docker [[1]]
        ValueError: Если название образа пустое

    Example:
        >>> check_and_pull_image("nvidia/cuda", "12.4.0-base")
        Pulling nvidia/cuda:12.4.0-base...
        [======>                             ] 25% Downloading...
        ...
        True
    """
    client = docker.from_env()
    full_image_name = f"{image_name}:{tag}"
    BAR_WIDTH = 120

    try:
        client.images.get(full_image_name)
        print(f"Образ {full_image_name} уже существует")
        return True
    except ImageNotFound:
        pass

    try:
        pull_log = client.api.pull(image_name, tag=tag, stream=True, decode=True)
        layer_bars = {}
        last_status = None

        for line in pull_log:
            if "id" not in line:
                continue

            layer_id = line["id"]
            status = line.get("status", "N/A")

            # Создаем/обновляем прогресс-бар
            if layer_id not in layer_bars:
                layer_bars[layer_id] = tqdm(
                    desc=f"{layer_id[:12]:<12}",
                    unit="B",
                    unit_scale=True,
                    leave=False,
                    ncols=BAR_WIDTH,
                    position=len(layer_bars),
                    dynamic_ncols=False,
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )

            bar = layer_bars[layer_id]
            if "progressDetail" in line:
                detail = line["progressDetail"]
                if "current" in detail and "total" in detail:
                    bar.total = detail["total"]
                    bar.update(detail["current"] - bar.n)

            # Обновляем статус
            if status != last_status:
                last_status = status
                postfix = f" {status}"[: BAR_WIDTH // 3]
                bar.set_postfix_str(postfix, refresh=False)

        # Завершаем все бары
        for bar in layer_bars.values():
            bar.close()

        print(f"\nDocker-образ {full_image_name} успешно загружен")
        return True

    except APIError as e:
        print(f"Ошибка при загрузке образа: {e}")
        return False
    finally:
        for bar in getattr(layer_bars, "values", []):
            bar.close()


def run_container(
    image_name: str,
    volumes: Dict[str, str],
    script_path: str,
    packages_to_install: Optional[List[str]] = None,
    use_gpu: bool = False,
    keep_container: bool = False,
    environment: Optional[Dict[str, str]] = None,
) -> None:
    """Запускает Docker-контейнер, монтирует папки, устанавливает пакеты, подключает GPU (если нужно)
    и выполняет Python-скрипт с возможностью передачи переменных окружения.

    Вывод контейнера, включая tqdm, отображается в реальном времени.

    Args:
        image_name (str): Имя Docker-образа.
        volumes (Dict[str, str]): Словарь, осуществляющий маппинг директорий, которые будут примонтированы
            к запущенному Docker-контейнеру.
            Ключи — пути на хосте, значения — пути внутри контейнера.
        script_path (str): Путь к Python-скрипту внутри контейнера, который будет исполняться.
        packages_to_install (Optional[List[str]]): Список Python-пакетов для установки (например, ["numpy", "pandas"]).
            По умолчанию None.
        use_gpu (bool): Флаг, указывающий, нужно ли подключать GPU. По умолчанию False.
        keep_container (bool): Флаг, указывающий, нужно ли оставлять контейнер запущенным после выполнения функции.
            По умолчанию False.
        environment (Optional[Dict[str, str]]): Словарь переменных окружения, которые будут переданы в контейнер.
            Например, {"HUGGING_FACE_TOKEN": "your_token_here"}. По умолчанию None.

    Raises:
        Exception: Если произошла ошибка при запуске контейнера или выполнении скрипта.

    Example:
        volumes = {
            "/host/path/data": "/container/path/data",
            "/host/path/scripts": "/container/path/scripts",
        }
        environment = {
            "HUGGING_FACE_TOKEN": "your_token_here",
            "OTHER_ENV_VAR": "value",
        }
        run_container(
            image_name="python:3.10",
            volumes=volumes,
            script_path="/container/path/scripts/run.py",
            packages_to_install=["numpy", "pandas"],
            use_gpu=True,
            keep_container=False,
            environment=environment,
        )
    """
    # Создание клиента Docker
    client = docker.from_env()

    # Определение запроса на использование GPU (если use_gpu=True)
    device_requests = []
    if use_gpu:
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    # Формирование volumes для монтирования папок
    volumes_dict = {
        host_path: {"bind": container_path, "mode": "rw"}  # 'rw' - чтение и запись
        for host_path, container_path in volumes.items()
    }

    # Формирование команды для установки пакетов и запуска скрипта
    install_cmd = (
        f"pip install {' '.join(packages_to_install)}" if packages_to_install else ""
    )
    scrypt_run_cmd = f"python -u {script_path}"
    interactive_shell_cmd = "exec bash" if keep_container else ""

    command = [install_cmd, scrypt_run_cmd, interactive_shell_cmd]
    command = list(filter(lambda x: x != "", command))
    command = " && ".join(command)

    # Объединение всех команд в одну
    command = f"sh -c '{command}'"

    # Вывод команды для отладки
    print(command)

    # Запуск контейнера
    container = client.containers.run(
        image_name,
        command=command,
        detach=True,
        stdout=True,
        stderr=True,
        tty=True,
        volumes=volumes_dict,
        device_requests=device_requests,
        environment=environment,  # Передача переменных окружения
    )

    # Чтение вывода контейнера в реальном времени
    print(f"Запущен контейнер с ID: {container.id}")
    try:
        for line in container.attach(stream=True, logs=True):
            print(line.decode("utf-8", errors="replace").strip())
    except KeyboardInterrupt:
        print("Остановлено пользователем.")

    # Остановка контейнера, если keep_container=False
    if not keep_container:
        container.remove(force=True)
        print(f"Контейнер {container.id} удален.")
    else:
        print(f"Контейнер {container.id} оставлен запущенным.")
        print(
            f"Для подключения к контейнеру выполните: docker exec -it {container.id} bash"
        )


def load_env_vars(env_file: str = ".env") -> Dict[str, Any]:
    """Загружает переменные окружения из файла .env и возвращает их в виде словаря.

    Args:
        env_file (str): Путь к файлу .env. По умолчанию ".env".

    Returns:
        Dict[str, Any]: Словарь, где ключи — имена переменных окружения, а значения — их значения.

    Example:
        env_vars = load_env_vars()
        print(env_vars)
        # {
        #     "HUGGING_FACE_TOKEN": "your_token_here",
        #     "DATABASE_URL": "postgres://user:password@localhost:5432/mydatabase",
        #     "DEBUG": True
        # }

    Raises:
        FileNotFoundError: Если файл .env не найден.
    """
    # Загружаем переменные окружения из файла .env
    load_dotenv(env_file)

    # Преобразуем переменные окружения в словарь
    env_vars = {}
    for key, value in os.environ.items():
        env_vars[key] = value

    return env_vars
