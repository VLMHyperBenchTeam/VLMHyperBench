import json
from typing import Dict, Any


class ConfigManager:
    def __init__(self, cfg_path: str, default: bool = False) -> None:
        """Инициализирует ConfigManager управляющий конфигурацией VLMHyperbecnh.

        Args:
            cfg_path (str): Путь к файлу конфигурации
            default (bool, optional): Флаг для инициализации конфига по умолчанию.
                По умолчанию False.

        Raises:
            FileNotFoundError: Если файл не найден при default=False
        """
        self.cfg_path = cfg_path

        if default:
            self.cfg = {
                "data_dirs": {
                    "vlm_base": "vlmhyperbench/vlm_base.csv",
                    "datasets": "vlmhyperbench/Datasets",
                    "model_answers": "vlmhyperbench/ModelsAnswers",
                    "model_metrics": "vlmhyperbench/ModelsMetrics",
                    "prompt_collections": "vlmhyperbench/PromptCollections",
                    "system_prompts": "vlmhyperbench/SystemPrompts",
                    "reports": "vlmhyperbench/Reports",
                },
                "system_dirs": {
                    "cfg": "vlmhyperbench/cfg",
                    "bench_stages": "vlmhyperbench/bench_stages",
                    "model_cache": "vlmhyperbench/model_cache",
                    "wheels": "vlmhyperbench/wheels",
                },
            }
        else:
            self.cfg = self.read_config(cfg_path)

    def write_config(self) -> None:
        """Сохраняет конфигурацию в JSON файл.

        Сохраняет текущее состояние cfg в файл с отступами и поддержкой Unicode.
        Использует путь, указанный в self.cfg_path.

        Raises:
            IOError: При ошибках записи файла
        """
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=4)

    @staticmethod
    def read_config(cfg_path: str) -> Dict[str, Any]:
        """Читает конфигурацию из JSON файла

        Args:
            cfg_path (str): Путь к файлу конфигурации

        Returns:
            Dict[str, Any]: Словарь с конфигурационными данными

        Raises:
            FileNotFoundError: Если файл не существует
            JSONDecodeError: При некорректном формате JSON
        """
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
