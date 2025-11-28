"""
YOLO інференція з попередньою обробкою зображень

Підтримує:
 - одиночний файл або папку зображень
 - вказівку шляху до моделі
 - налаштування порогу конфіденції
 - попередню обробку (Гаус, контраст, яскравість)
 - збереження розмитих результатів з детекцією
"""
from pathlib import Path
import argparse
import sys
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="YOLO інференс з обробкою зображень")
    p.add_argument("--model", type=Path, default=Path("runs/detect/train/weights/best.pt"),
                   help="Шлях до натренованої моделі")
    p.add_argument("--source", type=Path, default=Path("image2.jpg"),
                   help="Файл зображення або папка з зображеннями")
    p.add_argument("--conf", type=float, default=0.25, help="Порог впевненості")
    p.add_argument("--out", type=Path, default=Path("runs/predict"), help="Папка для результатів")
    p.add_argument("--blur", type=int, default=5, help="Розмір ядра Гауса (має бути непарним)")
    p.add_argument("--contrast", type=float, default=1.2, help="Коефіцієнт контрасту (1.0 = без змін)")
    p.add_argument("--brightness", type=int, default=10, help="Збільшення яскравості (-100 до 100)")
    p.add_argument("--equalize", action="store_true", help="Використовувати CLAHE")
    p.add_argument("--save_all", action="store_true", help="Зберегти оригінальні, розмиті та результати")
    return p.parse_args()

def preprocess_image(image, blur_kernel=5, contrast=1.2, brightness=10, equalize=False):
    """Попередня обробка зображення"""
    if image is None or image.size == 0:
        return image
    
    # 1️⃣ Гаусів фільтр
    if blur_kernel > 1 and blur_kernel % 2 == 1:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        print(f"   ✓ Застосовано Гаусів фільтр (kernel={blur_kernel})")
    
    # 2️⃣ CLAHE
    if equalize:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        print(f"   ✓ Застосовано CLAHE")
    
    # 3️⃣ Контраст та яскравість
    if contrast != 1.0 or brightness != 0:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        print(f"   ✓ Контраст={contrast}, Яскравість={brightness}")
    
    return image

def save_image(image, path: Path, name: str):
    """Зберегти зображення"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    print(f"   💾 {name}: {path}")

def draw_boxes_on_image(image, result, model):
    """Намалювати боксы на розмитому зображенні"""
    boxes = result.boxes
    
    for box in boxes:
        try:
            # Координати боксу
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Клас і впевненість
            cls_val = box.cls[0] if hasattr(box.cls, "__len__") else box.cls
            conf_val = box.conf[0] if hasattr(box.conf, "__len__") else box.conf
            class_id = int(cls_val)
            confidence = float(conf_val)
            
            class_name = model.names[class_id] if class_id in model.names else str(class_id)
            
            # Кольори для різних класів
            colors = {
                0: (0, 255, 0),    # Зелений
                1: (255, 0, 0),    # Синій
                2: (0, 0, 255),    # Червоний
                3: (255, 255, 0),  # Синій + Зелений
                4: (255, 0, 255),  # Пурпурний
            }
            color = colors.get(class_id % len(colors), (0, 255, 0))
            
            # Рисуємо прямокутник
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Рисуємо текст з фоном
            label = f"{class_name} {confidence:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 10
            
            # Фон для тексту
            cv2.rectangle(image, 
                         (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         color, -1)
            
            # Текст
            cv2.putText(image, label, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        except Exception as e:
            print(f"   ⚠️ Помилка при рисуванні боксу: {e}")
    
    return image

def main():
    args = parse_args()

    if not args.model.exists():
        print(f"❌ Модель не знайдена: {args.model}")
        sys.exit(1)

    model = YOLO(str(args.model))
    print(f"✅ Модель завантажена: {args.model}\n")

    # Перевіримо параметри обробки
    if args.blur % 2 == 0:
        args.blur += 1

    sources = []
    if args.source.is_dir():
        sources = sorted([p for p in args.source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    else:
        sources = [args.source]

    if not sources:
        print(f"❌ Не знайдено зображень у: {args.source}")
        sys.exit(1)

    print(f"📊 Параметри обробки:")
    print(f"   Гаусів фільтр: {args.blur}x{args.blur}")
    print(f"   Контраст: {args.contrast}")
    print(f"   Яскравість: {args.brightness}")
    print(f"   CLAHE: {'ТАК' if args.equalize else 'НІ'}\n")

    for idx, src in enumerate(sources, 1):
        print(f"🔍 Обробка ({idx}/{len(sources)}): {src.name}")
        
        # Завантажимо оригінальне зображення
        img_original = cv2.imread(str(src))
        if img_original is None:
            print(f"   ❌ Не вдалося завантажити: {src}")
            continue
        
        print(f"   📐 Розмір: {img_original.shape}")
        
        # Попередня обробка
        img_processed = preprocess_image(
            img_original.copy(),
            blur_kernel=args.blur,
            contrast=args.contrast,
            brightness=args.brightness,
            equalize=args.equalize
        )
        
        # Інференція на оброблених зображенні
        print(f"   🤖 Виконую детекцію...")
        results = model.predict(source=str(src), conf=args.conf, save=False, verbose=False)

        for i, result in enumerate(results):
            boxes = result.boxes
            n = len(boxes)
            print(f"   ✅ Знайдено об'єктів: {n}")

            if n > 0:
                for j, box in enumerate(boxes, 1):
                    cls_val = box.cls[0] if hasattr(box.cls, "__len__") else box.cls
                    conf_val = box.conf[0] if hasattr(box.conf, "__len__") else box.conf
                    class_id = int(cls_val)
                    confidence = float(conf_val)
                    try:
                        class_name = model.names[class_id]
                    except Exception:
                        class_name = str(class_id)
                    print(f"      [{j}] {class_name} — впевненість: {confidence:.2%}")

            # 🎨 Намалювати боксы на розмитому зображенні
            img_blurred_with_boxes = img_processed.copy()
            img_blurred_with_boxes = draw_boxes_on_image(img_blurred_with_boxes, result, model)
            
            # 💾 Збереження результатів
            out_stem = f"{src.stem}_blurred_pred"
            
            # Розмита фотка з боксами (ОСНОВНИЙ РЕЗУЛЬТАТ)
            blurred_path = args.out / f"{out_stem}{src.suffix}"
            save_image(img_blurred_with_boxes, blurred_path, "Розмита фотка з детекцією")
            
            # Опціонально: збереження оригіналу та розмитої без боксів
            if args.save_all:
                original_path = args.out / f"{src.stem}_original{src.suffix}"
                save_image(img_original, original_path, "Оригінальна фотка")
                
                processed_path = args.out / f"{src.stem}_processed{src.suffix}"
                save_image(img_processed, processed_path, "Розмита фотка (без боксів)")

    print("\n✨ Інференс завершено.")
    print(f"📁 Результати: {args.out}")

if __name__ == "__main__":
    main()
# ...existing code...