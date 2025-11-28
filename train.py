"""
YOLO Fine-tuning Script
Покращена версія з параметрами командного рядка, автодетекцією пристрою та базовою валідацією аргументів.
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Донавчання YOLOv8 на вашому датасеті")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Попередньо натренована модель або шлях до .pt")
    p.add_argument("--data", type=str, default="data/dataset.yaml", help="YAML файл датасету")
    p.add_argument("--epochs", type=int, default=30, help="Кількість епох")
    p.add_argument("--imgsz", type=int, default=640, help="Розмір зображення")
    p.add_argument("--batch", type=int, default=16, help="Розмір батча")
    p.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda:0' або індекс")
    p.add_argument("--project", type=str, default="runs/detect", help="Папка для збереження результатів")
    p.add_argument("--name", type=str, default="train", help="Ім'я експерименту")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--workers", type=int, default=None, help="Кількість робочих процесів (auto для GPU, 0 для CPU)")
    p.add_argument("--resume", action="store_true", help="Продовжити тренування, якщо є чекпоінти")
    return p.parse_args()

def choose_device(dev_arg: str) -> str:
    if dev_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev_arg

def validate_dataset(data_path: Path) -> bool:
    """Перевіримо, чи датасет існує і має правильну структуру"""
    if not data_path.exists():
        print(f"❌ Файл датасету не знайдено: {data_path}")
        return False
    
    data_dir = data_path.parent
    required_dirs = ["train/images", "valid/images", "test/images"]
    
    for req_dir in required_dirs:
        full_path = data_dir / req_dir
        if not full_path.exists():
            print(f"⚠️ Папка не знайдена: {full_path}")
    
    return True

def main():
    args = parse_args()

    data_path = Path(args.data)
    
    # Перевіримо датасет
    if not validate_dataset(data_path):
        print("❌ Датасет неправильно структурований!")
        print("Переконайтеся, що папка data/ містить:")
        print("  - dataset.yaml")
        print("  - train/images/")
        print("  - valid/images/")
        print("  - test/images/")
        sys.exit(1)

    device = choose_device(args.device)
    
    # Якщо workers не вказано, установимо автоматично
    workers = args.workers
    if workers is None:
        workers = 4 if device.startswith("cuda") else 0
    
    print(f"\n{'='*50}")
    print(f"✅ Використовую пристрій: {device}")
    print(f"✅ Модель: {args.model}")
    print(f"✅ Датасет: {data_path}")
    print(f"✅ Параметри: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, workers={workers}")
    print(f"{'='*50}\n")

    try:
        model = YOLO(args.model)
        print("✅ Модель завантажена успішно.")
    except Exception as e:
        print(f"❌ Не вдалося завантажити модель: {e}")
        sys.exit(1)

    try:
        print("🚀 Початок тренування...\n")
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            patience=args.patience,
            workers=workers,
            resume=args.resume,
            verbose=True
        )
        print("\n✨ Тренування завершено успішно!")
        print(f"📁 Результати збережено у: {args.project}/{args.name}/")
    except Exception as e:
        print(f"⚠️ Помилка під час тренування: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# ...existing code...