# 🍄 YOLO Fine-tuning — Детекція грибів

Короткий і актуальний README для цього проєкту: тренування, валідація та інференс моделей YOLO з прикладною попередньою обробкою зображень (Gaussian blur, CLAHE, контраст, яскравість).

## Огляд
Проєкт навчає і застосовує модель YOLO (Ultralytics) для детекції та розпізнавання видів грибів. Є скрипти:
- `train.py` — тренування (параметри через аргументи)
- `test.py` — валідація / тестування
- `predict.py` — інференс з попередньою обробкою (розмиття та збереження розмитих результатів)
- `requirements.txt` — залежності

## Класи грибів
Класи визначені в `data/dataset.yaml` (поле `names`). Поточні класи:
- Agaricus
- Boletus
- Cortinarius
- Entoloma
- Hygrocybe
- Lactarius
- Russula
- Suillus
- amanita

(Змінюйте `names:` у `data/dataset.yaml` за потреби.)

## Структура датасету
Очікувана структура:
```
data/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
Приклад `data/dataset.yaml` (рекомендується абсолютний або відносний `path`):
```yaml
path: C:\Users\MSI\Desktop\yolo-fine-tuning-mushroom\data
train: train/images
val: valid/images
test: test/images
nc: 9
names: ['Agaricus','Boletus','Cortinarius','Entoloma','Hygrocybe','Lactarius','Russula','Suillus','amanita']
```
Якщо Ultralytics шукає іншу директорію — перевірте `path:` та файл `%APPDATA%\Ultralytics\settings.json` або очистіть кеш `%APPDATA%\Ultralytics`.

## Встановлення залежностей
Рекомендується Python 3.11 (або сумісна версія). Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Якщо плануєте GPU (NVIDIA + CUDA), встановіть PyTorch з CUDA (приклад для CUDA 12.1):
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Перевірка:
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

## Як запускати
- Тренування:
```powershell
python train.py --epochs 30
# або з GPU
python train.py --epochs 30 --device cuda:0
```
Найкраща модель зберігається: `runs/detect/<name>/weights/best.pt`.

- Тестування/валідація:
```powershell
python test.py
```

- Інференс з попередньою обробкою:
```powershell
python predict.py --source image.jpg --model runs/detect/train/weights/best.pt --blur 7 --contrast 1.3 --brightness 10 --save_all
```
Опції (корисні):
- `--blur N` — Gaussian kernel (N непарне)
- `--equalize` — CLAHE
- `--save_all` — зберегти оригінал, оброблене і розмите з боксами

## Вихідні файли (predict)
За замовчуванням результати в `runs/predict/`:
- `{image}_blurred_pred.jpg` — розмита фотка з накладеними боксами (основний результат)
- `{image}_processed.jpg` — розмите зображення без боксів (якщо `--save_all`)
- `{image}_original.jpg` — оригінал (якщо `--save_all`)

## Поради та усунення неполадок
- Помилка `images not found`: перевірте `path:` у `data/dataset.yaml` і структуру папок.
- Якщо `ModuleNotFoundError: No module named 'torch'` — встановіть PyTorch у віртуальному середовищі.
- Якщо тренування йде на CPU замість GPU — перевстановіть PyTorch з CUDA і перевірте `nvidia-smi`.
- Щоб змусити застосувати Гаус до зображення і переконатися — використайте великий `--blur` (наприклад `21`) і `--save_all`, порівняйте `_processed` файл.

## Файли в репозиторії
- `train.py`, `test.py`, `predict.py`, `requirements.txt`, `data/dataset.yaml`, `README.md`, `runs/` (результати)

---

# ...existing code...