"""
YOLO Quick Test Script
Швидкий скрипт для тестування на окремих зображеннях
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def test_on_image(model, image_path, conf=0.25):
    """Тестує модель на одному зображенні"""
    print(f"\n📷 Обробляємо: {image_path}")

    # Прогноз
    results = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False
    )

    for result in results:
        print(f"   ✅ Виявлено об'єктів: {len(result.boxes)}")

        if len(result.boxes) > 0:
            print("   Деталі виявлень:")
            for idx, box in enumerate(result.boxes, 1):
                class_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                print(f"     {idx}. {class_name}")
                print(f"        Впевненість: {conf_score:.2%}")
                print(f"        Координати: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        else:
            print("   ℹ️  Об'єктів не виявлено")

    return results

def main():
    # Знайти найкращу модель
    model_path = Path('runs/detect/train/weights/best.pt')

    if not model_path.exists():
        print("❌ Модель не знайдена!")
        print("📝 Спочатку запустіть тренування:")
        print("   python train.py")
        return

    print(f"✅ Завантажена модель: {model_path}")
    print(f"🎯 Класи: BB (коричневі) та WB (білі)")

    # Завантажити модель
    model = YOLO(str(model_path))

    # Тестуємо на файлах з корневої папки (якщо є)
    test_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        test_files.extend(Path('.').glob(ext))

    if test_files:
        print(f"\n🔍 Знайдено {len(test_files)} зображень для тестування:")
        for img_file in test_files:
            test_on_image(model, img_file)

    # Тестуємо на тестовому наборі
    test_images_dir = Path('data/test/images')
    if test_images_dir.exists():
        test_images = sorted(list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png')))
        if test_images:
            print(f"\n\n📁 Тестування на {len(test_images)} зображеннях з data/test/images:")
            print("=" * 60)

            # Обробляємо перші 10 зображень
            for img_path in test_images[:10]:
                test_on_image(model, img_path)

    print("\n\n✨ Тестування завершено!")

if __name__ == '__main__':
    main()

