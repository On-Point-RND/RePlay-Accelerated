# Установка
Для запуска экспериментов скачайте Docker-образ:
    ```bash
    docker pull dmitryredkosk/bitsandbytes_recsys_clear
    ```

# Эксперименты

## Замена Classification Head с `nn.Linear` на `SwitchBackLinear`
В данном эксперименте производится замена слоя `nn.Linear` на слой `SwitchBackLinear`, который выполняет форвард-проходы в формате `int8`. Это позволяет ускорить обучение модели.

### Шаги для запуска

1. Перенесите содержимое файла `config_ml20_swichback.yaml` в основной файл конфигурации `config.yaml`

2. Запустите основной скрипт:
    ```bash
    python RePlay-Accelerated/main.py
    ```

## Замена `GELU` на `SeLU` и Classification Head на `nn.Linear`
В данном эксперименте производится замена функции активации `GELU` на функцию активации `SeLU`, а также замена 
Classification Head в блоке `_head` на `nn.Linear`. Это позволяет ускорить обучение модели. 

### Шаги для запуска

1. Перенесите содержимое файла `config_ml20_accelerate.yaml` в основной файл конфигурации `config.yaml`

2. Запустите основной скрипт:
    ```bash
    python RePlay-Accelerated/main.py
    ```

## Замена `GELU` на `SeLU` и Classification Head на `nn.Linear`
В данном эксперименте производится замена функции активации `GELU` на функцию активации `SeLU`, а также замена 
Classification Head в блоке `_head` на `nn.Linear`. Это позволяет ускорить обучение модели. 

### Шаги для запуска

1. Перенесите содержимое файла `config_ml20_accelerate.yaml` в основной файл конфигурации `config.yaml`

2. Запустите основной скрипт:
    ```bash
    python RePlay-Accelerated/main.py
    ```

## Обучение в `mixed-precision`
Для того чтобы запустить обучение со смешанной точности достаточно в поле `precision` в конфигурационном файле `/configs/model/bert4rec_movielens_20m.yaml` указать `bf16-mixed`.
Это позволяет ускорить обучение модели. Однако, валидация модели во время обучения может замедлится вследствие особенностей `trainer` из pytorch lightning.

### Шаги для запуска

1. в конфигурационном файле `/configs/model/bert4rec_movielens_20m.yaml` указать:
    ```
    precision: bf16-mixed
    ```

2. Запустите основной скрипт:
    ```bash
    python RePlay-Accelerated/main.py
    ```

## Проверка обновления слоя

Чтобы убедиться, что функции активация `GELU` была заменена на `SeLU`, или слой `SwitchBackLinear` успешно заменил `nn.Linear`, вы можете добавить вывод инициализированной модели в метод `run` файла `RePlay-Accelerated/replay_benchmarks/train_runner.py`:
```python
    def run(self):
        """Execute the training pipeline."""
        train_dataloader, val_dataloader, prediction_dataloader = (
            self._load_dataloaders()
        )
        logging.info("Initializing model...")
        model = self._initialize_model()
        print(model)
```