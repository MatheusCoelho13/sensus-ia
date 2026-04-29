"""Script para avaliar modelo YOLOv8 treinado."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Avaliate YOLOv8 model")
    parser.add_argument('--weights', default='../runs/detect/custom_run/weights/best.pt', help='path to weights')
    parser.add_argument('--data', default='../config/data.yaml', help='path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    
    args = parser.parse_args()
    
    print(f"Carregando modelo: {args.weights}")
    model = YOLO(args.weights)
    
    print(f"Avaliando em: {args.data}")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch
    )
    
    print("\n✅ Validação concluída!")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")


if __name__ == '__main__':
    main()
