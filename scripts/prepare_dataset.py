"""Limpa dataset: remove imagens corrompidas e gera labels automáticas com YOLOv8."""
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2

def validate_images(dataset_dir='../datasets'):
    """Remove imagens corrompidas."""
    dataset_path = Path(dataset_dir)
    removed = 0
    
    img_files = list(dataset_path.glob('images/*/*.jpg')) + \
                list(dataset_path.glob('images/*/*.png'))
    for img_path in img_files:
        try:
            img = Image.open(img_path)
            img.verify()
            # verificar tamanho (muito grande = problema)
            w, h = img.size
            if w * h > 200000000:  # > 200M pixels
                print(f"Removendo imagem gigante: {img_path} ({w}x{h})")
                img_path.unlink()
                removed += 1
        except Exception as e:
            print(f"Removendo imagem corrompida: {img_path} -> {e}")
            img_path.unlink()
            removed += 1
    
    print(f"✓ Removidas {removed} imagens corrompidas")

def generate_labels(dataset_dir, model_path='yolov8n.pt'):
    """Gera labels automáticas usando modelo pré-treinado."""
    model = YOLO(model_path)
    dataset_path = Path(dataset_dir)
    
    # criar pastas de labels
    labels_train = dataset_path / 'labels' / 'train'
    labels_val = dataset_path / 'labels' / 'val'
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    
    # gerar labels para cada split
    for split in ['train', 'val']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        img_files = list(images_dir.glob('*/*.jpg')) + list(images_dir.glob('*/*.png'))
        
        for i, img_path in enumerate(img_files):
            print(f"[{split}] Processando {i+1}/{len(img_files)}: {img_path.name}")
            
            try:
                # inferência
                results = model.predict(str(img_path), conf=0.25, verbose=False)
                
                # extrair boxes em formato YOLO
                r = results[0]
                label_path = labels_dir / img_path.parent.name / (img_path.stem + '.txt')
                label_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(label_path, 'w') as f:
                    if len(r.boxes) > 0:
                        # imagem h, w
                        h, w = r.orig_shape
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            # box.xywhn = normalized [x_center, y_center, width, height]
                            xywhn = box.xywhn[0]
                            f.write(f"{cls} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")
                        print(f"  ✓ {len(r.boxes)} objetos detectados")
                    else:
                        print(f"  ⊘ nenhum objeto detectado")
            except Exception as e:
                print(f"  ✗ erro ao processar {img_path.name}: {e}")
                continue
    
    print("✓ Labels gerados!")

if __name__ == '__main__':
    dataset_dir = '../datasets'
    print("Limpando imagens corrompidas...")
    validate_images(dataset_dir)
    
    print("\nGerando labels automáticas...")
    generate_labels(dataset_dir)
    
    print("\n✅ Dataset pronto para treino!")
