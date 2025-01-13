# Описание

В данном репозитории показано использование модулей
* subprocess (`src/pipeline_subprocess.py`)
* docker (`src/pipeline_docker-sdk.py`)

для запуска тестового пайплайна с docker-контейнерами из двух шагов:

Шаг 1.

Запускается Docker-контейнер указанный в переменной `image_name`.

Данный контейнер выполняет скрипт `pipeline/scripts/writer.py` без поддержи ГПУ.

Шаг 2.

Запускается Docker-контейнер указанный в переменной `image_name`.

Данный контейнер выполняет скрипт `pipeline/scripts/reader.py` с поддержкой ГПУ.

При этом в терминал из которого был запущен пайплайн на исполнение, так же выводятся все логи внутри docker-контейнера, что очень удобно для отладки и анализа.

Ниже приведен вывод терминала, исполняющего пайплайн (`src/pipeline_docker-sdk.py`):
```
Запущен контейнер с ID: 58758b38449149c2f91cef424f5203b9bca738c65dece9ee22aa0548c7e2d818
==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .
sh: 1: nvidia-smi: not found
Processing:   0% 0/10 [00:00<?, ?iteration/s]
Result: 0
Processing:  10% 1/10 [00:01<00:09,  1.00s/iteration]
Result: 1
Processing:  20% 2/10 [00:02<00:08,  1.00s/iteration]
Result: 2
Processing:  30% 3/10 [00:03<00:07,  1.00s/iteration]
Result: 3
Processing:  40% 4/10 [00:04<00:06,  1.00s/iteration]
Result: 4
Processing:  50% 5/10 [00:05<00:05,  1.00s/iteration]
Result: 5
Processing:  60% 6/10 [00:06<00:04,  1.00s/iteration]
Result: 6
Processing:  70% 7/10 [00:07<00:03,  1.00s/iteration]
Result: 7
Processing:  80% 8/10 [00:08<00:02,  1.00s/iteration]
Result: 8
Processing:  90% 9/10 [00:09<00:01,  1.00s/iteration]
Result: 9
Processing: 100% 10/10 [00:10<00:00,  1.00s/iteration]
Контейнер 58758b38449149c2f91cef424f5203b9bca738c65dece9ee22aa0548c7e2d818 удален.
Запущен контейнер с ID: d9a94eaf055b880fcbb2d1eeaafce7e16b0ed9c36d971c4f94337462fd3e1a49
==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
Sun Jan 12 11:49:11 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.01             Driver Version: 535.216.01   CUDA Version: 12.4     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A10                     On  | 00000000:01:00.0 Off |                  Off |
|  0%   26C    P8              15W / 150W |      0MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
torch.__version__ 2.1.2+cu121
torch.cuda.is_available() True
torch.cuda.current_device() 0
torch.cuda.device_count() 0
torch.cuda.get_device_name(0) NVIDIA A10
Processing lines: 0line [00:00, ?line/s]
Result: 0
Processing lines: 1line [00:01,  1.00s/line]
Result: 2
Processing lines: 2line [00:02,  1.00s/line]
Result: 4
Processing lines: 3line [00:03,  1.00s/line]
Result: 6
Processing lines: 4line [00:04,  1.00s/line]
Result: 8
Processing lines: 5line [00:05,  1.00s/line]
Result: 10
Processing lines: 6line [00:06,  1.00s/line]
Result: 12
Processing lines: 7line [00:07,  1.00s/line]
Result: 14
Processing lines: 8line [00:08,  1.00s/line]
Result: 16
Processing lines: 9line [00:09,  1.00s/line]
Result: 18
Processing lines: 10line [00:10,  1.00s/line]
Контейнер d9a94eaf055b880fcbb2d1eeaafce7e16b0ed9c36d971c4f94337462fd3e1a49 удален.
```

# Сравнение данных подходов с точки зрения безопасности

Использование Docker API (например, через `docker-py`) считается более безопасным, чем использование `subprocess` для вызова команд Docker, по нескольким причинам:

---

### 1. **Избежание инъекций команд**
   При использовании `subprocess` вы передаете команды как строки или списки аргументов. Если входные данные не проверяются должным образом, это может привести к **инъекциям команд**. Например:

   ```python
   user_input = "my_image; rm -rf /"  # Злонамеренный ввод
   subprocess.run(f"docker run {user_input}", shell=True)  # Опасный вызов
   ```

   Docker API, напротив, работает через вызовы методов и не подвержен этой уязвимости, так как параметры передаются в виде структурированных данных, а не строк.

   ```python
   client.containers.run("my_image")  # Безопасно, даже если image_name содержит спецсимволы
   ```

---

### 2. **Контроль над параметрами**
   Docker API предоставляет строго типизированные методы и параметры, что снижает вероятность ошибок. Например, при использовании `subprocess` вы можете случайно передать неправильные флаги или опции, что может привести к неожиданному поведению или уязвимостям.

   С Docker API вы явно указываете параметры, и они проверяются на этапе выполнения:

   ```python
   client.containers.run(
       image="my_image",
       detach=True,
       mem_limit="512m",
       network_mode="none"
   )
   ```

   Это исключает возможность опечаток или неправильной интерпретации параметров.

---

### 3. **Избежание проблем с экранированием**
   При использовании `subprocess` вам нужно вручную экранировать аргументы, чтобы избежать проблем с пробелами, кавычками и другими специальными символами. Docker API избавляет вас от этой необходимости, так как он сам обрабатывает передачу параметров.

---

### 4. **Лучшая интеграция с окружением**
   Docker API использует те же механизмы аутентификации и авторизации, что и Docker CLI, но делает это более безопасно. Например, если Docker настроен на использование TLS для защиты API, `docker-py` автоматически подключается с использованием этих настроек. В случае с `subprocess` вам придется вручную управлять сертификатами и ключами.

---

### 5. **Упрощение управления ошибками**
   Docker API предоставляет встроенные механизмы обработки ошибок, которые помогают избежать непредвиденных ситуаций. Например, если контейнер не может быть запущен, `docker-py` выбросит исключение, которое можно легко обработать.

   ```python
   try:
       client.containers.run("nonexistent_image")
   except docker.errors.ImageNotFound:
       print("Образ не найден")
   ```

   В случае с `subprocess` вам придется вручную анализировать вывод и коды возврата, что увеличивает вероятность ошибок.

---

### 6. **Избежание shell-инъекций**
   Если вы используете `subprocess` с `shell=True`, вы подвергаетесь риску shell-инъекций, так как команда выполняется через оболочку (например, `/bin/sh`). Docker API не использует оболочку, поэтому этот риск исключен.

---

### 7. **Более безопасное управление процессами**
   Docker API предоставляет более высокоуровневый интерфейс для управления контейнерами, что снижает вероятность ошибок, связанных с неправильным использованием низкоуровневых команд. Например, вы можете легко управлять жизненным циклом контейнера, сетями, томами и другими ресурсами, не вызывая отдельные команды через `subprocess`.

---

### 8. **Поддержка современных функций Docker**
   Docker API всегда поддерживает последние функции Docker, такие как управление ресурсами, сети, секреты и т.д. При использовании `subprocess` вам придется вручную обновлять команды, чтобы поддерживать новые функции, что может привести к ошибкам.

---

### Когда `subprocess` может быть приемлемым?
Использование `subprocess` может быть оправдано в простых сценариях, где:
- Вы полностью контролируете входные данные.
- Вы не используете `shell=True`.
- Вы тщательно проверяете и экранируете все аргументы.
- Вам не нужна сложная логика управления контейнерами.

---

### Итог
Docker API безопаснее, чем `subprocess`, потому что он:
- Исключает инъекции команд.
- Предоставляет строгую типизацию и проверку параметров.
- Упрощает управление ошибками.
- Интегрируется с современными функциями Docker.
- Избегает проблем с экранированием и shell-инъекциями.

Если вы работаете с Docker в Python, использование `docker-py` — это более безопасный и удобный подход.