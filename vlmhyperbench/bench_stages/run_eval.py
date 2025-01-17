import time

from some_package.my_module import Calculator, greet
from tqdm import tqdm


if __name__ == "__main__":
    # Имя файла, из которого будем читать данные
    input_txt = "data/task1_output.txt"
    output_txt = "data/task2_output.txt"

    # Тест модуля добавленного в докер-контейнер
    print("Тестируем some_package")
    print("-" * 30)
    print(greet(name="Ivan"))
    calc = Calculator()
    print(calc.add(1, 3))
    print("-" * 30)

    with open(input_txt, "r") as input_file, open(output_txt, "w") as output_file:
        # Читаем файл построчно
        for line in tqdm(input_file, desc="Processing lines", unit="line"):
            number = int(line.strip())
            doubled_number = number * 2

            print(f"\nResult: {doubled_number}")
            output_file.write(f"{doubled_number}\n")
            output_file.flush()

            # Ждем одну секунду
            time.sleep(1)
