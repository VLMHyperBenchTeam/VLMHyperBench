from dataclasses import dataclass


@dataclass
class BenchmarkRunConfig:
    """
    Dataclass для хранения всех параметров запуска бенчмарка для одной модели в рамках одной структуры с полями.

    Attributes:
        dataset (str): Название датасета, на котором будем оценивать работу модели.
        framework (str): Название фреймворка для инференса VLM-моделей (например, "Hugging Face", "vLLM", "SgLang").
        model_family (str): Семейство VLM-модели, которую будем оценивать на бенчмарке (например, "Qwen2-VL").
        model_name (str): Название VLM-модели, которую будем оценивать на бенчмарке (например, "Qwen2-VL-2B").
        docker_image (str | None): Название Docker image, в котором будем проводить бенчмарк модели. Optional.
            Defaults to None.
            Если None, то будет использован Docker image из `vlmhyperbench/vlm_base.csv`,
            для запуска указанной модели в указанном фреймворке инференса.
        system_prompt (str | None): Название текстового файла, содержащего system prompt,
            который будет передан VLM-модели при ее инициализации. Optional.
            Defaults to None.
        prompt_collection (list[str]): Название csv-файла, содержащего коллекцию промптов для модели,
            которые и будут использованы при бенчмарке модели.
        metrics (list[str] | None): Список метрик, которые будем оценивать по ответам модели
            (например, ['WER', 'CER', 'BLEU']). Optional. Defaults to None.
            Если None, то метрики оцениваться не будут и этап 2 будет пропущен.
        only_evaluate_metrics (bool): Если True, то пропускаем этап 1 оценки VLM-модели на датасете
            и сразу переходим к этапу оценки метрик. Optional.
            Defaults to False.
        metrics_aggregators (list[str] | None): Список типов агрегаторов, используя которые рассчитываем метрики
            (например, ["by_id", "by_doc_type", "overall"]). Optional. Defaults to None.
            Если None, то метрики оцениваться не будут и этап 2 будет пропущен.
        filter_doc_class (str | None): Разметка датасета annotation.csv, будет отфильтрована, так чтобы
            в датасете в столбце "doc_class" остались только значения равные заданному. Optional. Defaults to None.
        filter_question_type (str | None): Разметка датасета annotation.csv, будет отфильтрована, так чтобы
            в датасете в столбце "question_type" остались только значения равные заданному. Optional. Defaults to None.
    """

    dataset: str
    framework: str
    model_family: str
    model_name: str
    prompt_collection: list[str]
    docker_image: str | None = None
    system_prompt: str | None = None
    metrics: list[str] | None = None
    only_evaluate_metrics: bool = False
    metrics_aggregators: list[str] | None = None
    filter_doc_class: str | None = None
    filter_question_type: str | None = None


if __name__ == "__main__":
    # Пример использования
    config = BenchmarkRunConfig(
        dataset="Passport_MINI",
        framework="Hugging Face",
        model_family="Qwen2-VL",
        model_name="Qwen2-VL-2B",
        prompt_collection="AntonShiryaev.csv",
    )

    print(config)
