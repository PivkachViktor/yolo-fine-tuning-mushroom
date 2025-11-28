# ...existing code...
"""
Скрипт тестування моделі YOLO на тестовому наборі.
Підтримує:
 - автоматичний пошук останньої моделі в runs/detect/train*/weights/best.pt
 - валідацію на split='test'
 - детектування на кількох тестових зображеннях і збереження результатів
 - налаштування через аргументи командного рядка
"""
from pathlib import Path
import argparse
import sys
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Тестування натренованої YOLO моделі")
    p.add_argument("--runs", type=Path, default=Path("runs/detect"),
                   help="Папка з результатами тренувань (за замовчуванням: runs/detect)")
    p.add_argument("--data", type=Path, default=Path("data/dataset.yaml"),
                   help="YAML з конфігурацією датасету")
    p.add_argument("--conf", type=float, default=0.25, help="Порог впевненості для валідації/інференсу")
    p.add_argument("--iou", type=float, default=0.6, help="IOU для метрик валідації")
    p.add_argument("--save_project", type=Path, default=Path("runs/test_predictions"),
                   help="Папка для збережених передбачень")
    p.add_argument("--max_images", type=int, default=5, help="Макс. кількість тестових зображень для інференсу")
    return p.parse_args()

def find_latest_model(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")])
    if not train_dirs:
        return None
    candidate = train_dirs[-1] / "weights" / "best.pt"
    return candidate if candidate.exists() else None

def print_val_results(results):
    # Результат може мати атрибути, залежно від версії ultralytics
    try:
        box = results.box
    except Exception:
        box = getattr(results, "boxes", None)
    try:
        map50 = getattr(results.box, "map50", None) or getattr(results, "map50", None)
        map_all = getattr(results.box, "map", None) or getattr(results, "map", None)
        precision = getattr(results.box, "p", None)
        recall = getattr(results.box, "r", None)
        print("\n📊 Результати валідації:")
        if map50 is not None:
            print(f"mAP50: {map50:.4f}")
        if map_all is not None:
            print(f"mAP50-95: {map_all:.4f}")
        if precision is not None:
            try:
                print(f"Precision: {precision.mean():.4f}")
            except Exception:
                print(f"Precision: {precision}")
        if recall is not None:
            try:
                print(f"Recall: {recall.mean():.4f}")
            except Exception:
                print(f"Recall: {recall}")
    except Exception:
        print("⚠️ Не вдалося вивести детальні метрики (залежить від версії ultralytics).")

def run_inference_on_images(model: YOLO, images_dir: Path, project: Path, name: str, conf: float, max_images: int):
    if not images_dir.exists():
        print(f"❌ Тестова папка не знайдена: {images_dir}")
        return
    imgs = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        print(f"❌ Не знайдено зображень у: {images_dir}")
        return
    print(f"\n🖼️ Знайдено {len(imgs)} зображень, обробимо перші {min(len(imgs), max_images)}")
    project.mkdir(parents=True, exist_ok=True)
    for img in imgs[:max_images]:
        print(f"  → Обробляю: {img.name}")
        results = model.predict(
            source=str(img),
            conf=conf,
            save=True,
            project=str(project),
            name=name
        )
        for res in results:
            boxes = getattr(res, "boxes", [])
            if boxes:
                print(f"    Виявлено {len(boxes)} об'єктів:")
                for b in boxes:
                    try:
                        cls_id = int(b.cls[0])
                        conf_val = float(b.conf[0])
                        cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
                        print(f"      - {cls_name}: {conf_val:.2%}")
                    except Exception:
                        print("      - (інформація про коробку недоступна)")
            else:
                print("    Об'єктів не виявлено")

def main():
    args = parse_args()

    model_path = find_latest_model(args.runs)
    if model_path is None:
        print("❌ Не знайдено натренованої моделі в runs/detect/*/weights/best.pt")
        print("Запустіть спочатку тренування: python train.py")
        sys.exit(1)

    print(f"✅ Використовую модель: {model_path}")
    model = YOLO(str(model_path))

    # Виконати валідацію (split='test')
    print("\n🔍 Виконую валідацію на тестовому наборі...")
    try:
        val_res = model.val(data=str(args.data), split="test", conf=args.conf, iou=args.iou)
        print_val_results(val_res)
    except Exception as e:
        print(f"⚠️ Помилка під час валідації: {e}")

    # Інференс на прикладних тестових зображеннях
    test_images_dir = Path("data/test/images")
    run_inference_on_images(model, test_images_dir, args.save_project, name="results", conf=args.conf, max_images=args.max_images)

    print("\n✨ Тестування завершено.")
    print(f"📁 Перегляньте збережені передбачення у: {args.save_project / 'results'}")

if __name__ == "__main__":
    main()
# ...existing code...