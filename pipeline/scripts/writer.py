import os
import time

from tqdm import tqdm

from some_package.my_module import Calculator, greet


if __name__ == "__main__":
    # Имя файла, в который будем записывать данные
    filename = "data/task1_output.txt"

    # Тест модуля добавленного в докер-контейнер
    print("Тестируем some_package")
    print("-"*30)
    print(greet(name="Ivan"))
    calc = Calculator()
    print(calc.add(1, 3))
    print("-"*30)

    os.system("nvidia-smi")

    with open(filename, "w") as file:
        for i in tqdm(range(10), desc="Processing", unit="iteration"):
            line = i
            file.write(f"{i}\n")
            print(f"\nResult: {line}")
            file.flush()

            # Ждем одну секунду
            time.sleep(1)
