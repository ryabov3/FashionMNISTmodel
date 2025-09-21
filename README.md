# Модель FashionMNIST

Этот проект реализует сверточную нейронную сеть (CNN) для классификации изображений из набора данных FashionMNIST с использованием PyTorch. Модель достигает точности **93.5%** на тестовой выборке и работает исключительно на CPU.

## Структура проекта

```
.
├── api.py              # FastAPI-приложение для инференса модели
├── train_model.py      # Скрипт для обучения модели
├── test_model.py       # Скрипт для тестирования модели
├── model.py            # Определение архитектуры FashionMNISTModel
├── Dockerfile          # Dockerfile для сборки контейнера
├── models/             # Директория для хранения обученных моделей
├── fashion_mnist/      # Набор данных FashionMNIST
├── uv.lock             # Lock-файл с зависимостями
├── pyproject.toml      # Зависимости проекта
└── README.md           # Документация проекта
```

## Архитектура модели

Модель `FashionMNISTModel` — это сверточная нейронная сеть, реализованная в PyTorch со следующей архитектурой:

```python
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, bias=False),  # 26x26
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, bias=False),  # 24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),  # 10x10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 5x5
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(64*5*5, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```

## Зависимости

Проект использует следующие Python-пакеты, указанные в `pyproject.toml`:

- fastapi>=0.116.2
- loguru>=0.7.3
- matplotlib>=3.10.6
- numpy>=2.3.3
- pandas>=2.3.2
- pillow>=11.3.0
- python-multipart>=0.0.20
- torch>=2.8.0
- torchvision>=0.23.0
- tqdm>=4.67.1
- uvicorn>=0.35.0

Зависимости управляются с помощью пакетного менеджера `uv`. Для их установки выполните следующие шаги:

1. Установите `uv` (если он еще не установлен):
   ```bash
   pip install uv
   ```

2. Установите зависимости:
   ```bash
   uv sync
   ```

Это установит все необходимые пакеты, указанные в `uv.lock`.

## Настройка Docker

Для сборки и запуска Docker-контейнера выполните следующие шаги:

1. **Сборка Docker-образа**:
   ```bash
   docker build -t fashionmnist-model .
   ```

2. **Запуск Docker-контейнера**:
   FastAPI-приложение работает на порту 8000 внутри контейнера. Пробросьте его на порт 8080 хоста:
   ```bash
   docker run -d -p 8080:8000 fashionmnist-model
   ```

Это запустит контейнер в фоновом режиме, и API будет доступен по адресу `http://localhost:8080`.

## Использование

1. **Обучение модели**:
   Запустите скрипт для обучения модели на наборе данных FashionMNIST:
   ```bash
   python train_model.py
   ```

2. **Тестирование модели**:
   Оцените модель на тестовой выборке:
   ```bash
   python test_model.py
   ```

3. **Запуск API**:
   Запустите FastAPI-сервер для инференса:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

   Или используйте Docker-контейнер, как описано выше.

## Примечания

- Модель разработана для работы на CPU, что обеспечивает совместимость с системами без GPU.
- Набор данных FashionMNIST автоматически загружается в директорию `fashion_mnist/` при запуске скрипта обучения.
- Обученные модели сохраняются в директории `models/` для повторного использования в инференсе.