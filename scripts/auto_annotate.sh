#!/bin/bash
# Script para gerar anotações YOLO automaticamente usando um modelo pré-treinado
# Uso: bash scripts/auto_annotate.sh

echo "🤖 Gerando anotações automáticas para dataset custom..."
echo ""

VENV_PY=".venv/Scripts/python"
if [ ! -f "$VENV_PY" ]; then
    VENV_PY=".venv/bin/python"
fi

# Verificar se tem PyTorch
if ! $VENV_PY -c "import torch; print('PyTorch OK')" &>/dev/null; then
    echo "❌ PyTorch não encontrado. Execute: make setup"
    exit 1
fi

echo "📁 Criando pastas de labels..."
mkdir -p datasets/labels/train/cadeira
mkdir -p datasets/labels/train/pessoa
mkdir -p datasets/labels/val/cadeira
mkdir -p datasets/labels/val/mesa
mkdir -p datasets/labels/val/parede
mkdir -p datasets/labels/val/porta

echo ""
echo "🔄 Gerando anotações com YOLOv8..."

$VENV_PY << 'EOF'
from ultralytics import YOLO
from pathlib import Path
import os

# Carregar modelo pré-treinado
print("📥 Carregando modelo YOLOv8n pré-treinado...")
model = YOLO('yolov8n.pt')

# Estrutura do dataset
dataset_paths = {
    'datasets/images/train': {
        'cadeira': 0,
        'pessoa': 0,
    },
    'datasets/images/val': {
        'cadeira': 0,
        'mesa': 2,
        'parede': 3,
        'porta': 4,
    }
}

class_mapping = {
    'person': 0,
    'backpack': 1,
    'chair': 2,
    'bench': 3,
    'laptop': 4,
}

total = 0
processed = 0

# Contar total de imagens
for base_path, classes_dict in dataset_paths.items():
    for class_name in classes_dict.keys():
        class_path = Path(base_path) / class_name
        if class_path.exists():
            total += len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))

print(f"📊 Total de imagens para anotar: {total}")
print("")

# Processar cada classe
for base_path, classes_dict in dataset_paths.items():
    split = 'train' if 'train' in base_path else 'val'
    
    for class_name, class_id in classes_dict.items():
        class_path = Path(base_path) / class_name
        if not class_path.exists():
            continue
        
        # Procurar imagens
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        if not images:
            print(f"⚠️  Nenhuma imagem em {class_path}")
            continue
        
        print(f"🔍 Processando: {split}/{class_name} ({len(images)} imagens)")
        
        # Criar pasta de labels se não existir
        labels_dir = Path(f'datasets/labels/{split}/{class_name}')
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Processar cada imagem
        for img_path in images:
            try:
                # Detectar com modelo pré-treinado
                results = model.predict(source=str(img_path), conf=0.3, verbose=False)
                
                # Salvar anotações em formato YOLO
                label_path = labels_dir / (img_path.stem + '.txt')
                
                with open(label_path, 'w') as f:
                    for result in results:
                        for box in result.boxes:
                            # Obter classe detectada
                            cls = int(box.cls[0])
                            
                            # Normalizar para 0-1 (formato YOLO)
                            x_center = box.xywh[0][0] / result.orig_shape[1]
                            y_center = box.xywh[0][1] / result.orig_shape[0]
                            width = box.xywh[0][2] / result.orig_shape[1]
                            height = box.xywh[0][3] / result.orig_shape[0]
                            
                            # Clampar valores entre 0 e 1
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                processed += 1
                if processed % 5 == 0:
                    print(f"  ✅ {processed}/{total} imagens processadas")
                    
            except Exception as e:
                print(f"  ❌ Erro ao processar {img_path.name}: {e}")
        
        print("")

print(f"✅ Anotações geradas para {processed}/{total} imagens")
print("")
print("📁 Estrutura criada:")
print("  datasets/labels/train/cadeira/ ✅")
print("  datasets/labels/train/pessoa/ ✅")
print("  datasets/labels/val/cadeira/ ✅")
print("  datasets/labels/val/mesa/ ✅")
print("  datasets/labels/val/parede/ ✅")
print("  datasets/labels/val/porta/ ✅")
print("")
print("🚀 Pronto! Agora pode rodar: make start_train_cuda_custom")
EOF

echo ""
echo "✅ Concluído!"
