"""Script para detectar, desenhar, nomear e medir distância dos objetos."""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse


def calculate_distance(box_area, image_area, base_distance=2.0):
    """Estima distância baseada no tamanho do box.
    
    Premissa: quanto maior o objeto na imagem, mais perto está.
    distance = base_distance / sqrt(area_ratio)
    """
    area_ratio = box_area / image_area
    if area_ratio < 0.001:
        return "Muito distante (>5m)"
    elif area_ratio < 0.01:
        return "Distante (3-5m)"
    elif area_ratio < 0.05:
        return "Próximo (1-3m)"
    else:
        return "Muito próximo (<1m)"


def analyze_image(image_path, model_path='../models/yolov8n.pt', output_dir='../output'):
    """Detecta, desenha e mede distância dos objetos."""
    
    # Carregar modelo
    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Ler imagem
    print(f"Processando: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Erro ao carregar imagem: {image_path}")
        return
    
    h, w = img.shape[:2]
    image_area = h * w
    
    # Detecção
    results = model(img)
    r = results[0]
    
    # Informações de detecção
    detections = []
    
    if len(r.boxes) > 0:
        print(f"\n✅ {len(r.boxes)} objeto(s) detectado(s):\n")
        
        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            nome = model.names[cls]
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calcular área
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / image_area
            
            # Calcular distância
            distance = calculate_distance(box_area, image_area)
            
            # Centro do objeto
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Armazenar detecção
            detections.append({
                'nome': nome,
                'confiança': conf,
                'bbox': (x1, y1, x2, y2),
                'area_ratio': area_ratio,
                'distancia': distance,
                'centro': (cx, cy)
            })
            
            # Imprimir
            print(f"  [{i+1}] {nome.upper()}")
            print(f"       Confiança: {conf:.2%}")
            print(f"       Tamanho: {area_ratio:.2%} da imagem")
            print(f"       Distância: {distance}")
            print(f"       Posição: ({x1}, {y1}) -> ({x2}, {y2})")
            print()
            
            # Desenhar bounding box
            color = (0, 255, 0)  # Verde
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label com info
            label = f"{nome} {conf:.2%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background para texto
            cv2.rectangle(img, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1),
                         color, -1)
            
            # Texto
            cv2.putText(img, label, 
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Distância abaixo da bbox
            cv2.putText(img, distance,
                       (x1 + 5, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        print("⊘ Nenhum objeto detectado")
    
    # Salvar imagem com anotações
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = Path(image_path).stem
    output_file = output_path / f"{filename}_detected.jpg"
    
    cv2.imwrite(str(output_file), img)
    print(f"\n✅ Imagem salva: {output_file}")
    
    # Retornar detecções
    return detections


def analyze_file_or_camera(source, model_path='../models/yolov8n.pt', output_dir='../output'):
    """Analisa arquivo, URL ou câmera."""
    
    if source.lower() in ('webcam', '0', 'camera'):
        print("📷 Abrindo câmera...")
        cap = cv2.VideoCapture(0)
        
        model = YOLO(model_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecção em tempo real
            results = model(frame)
            r = results[0]
            
            h, w = frame.shape[:2]
            image_area = h * w
            
            # Desenhar boxes
            for box in r.boxes:
                cls = int(box.cls[0])
                nome = model.names[cls]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                box_area = (x2 - x1) * (y2 - y1)
                distance = calculate_distance(box_area, image_area)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{nome} {conf:.0%} - {distance}"
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Assistiva IA - Detecção em Tempo Real', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif source.startswith('http'):
        print(f"📥 Baixando imagem de URL: {source}")
        import requests
        resp = requests.get(source)
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Salvar temporariamente
        temp_path = Path('/tmp/temp_image.jpg')
        cv2.imwrite(str(temp_path), img)
        
        analyze_image(temp_path, model_path, output_dir)
    
    else:
        # Arquivo local
        analyze_image(source, model_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detectar e medir distância de objetos')
    parser.add_argument('source', default='webcam', nargs='?',
                       help='Arquivo de imagem, URL ou "webcam" para câmera em tempo real')
    parser.add_argument('--model', default='../models/yolov8n.pt',
                       help='Caminho para o modelo (.pt)')
    parser.add_argument('--output', default='../output',
                       help='Diretório de saída')
    
    args = parser.parse_args()
    
    analyze_file_or_camera(args.source, args.model, args.output)
