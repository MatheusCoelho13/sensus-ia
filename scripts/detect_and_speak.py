"""Script para detectar objetos, desenhar hitbox, nomear e FALAR os objetos."""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import requests
import pyttsx3
import threading


class SpeakThread(threading.Thread):
    """Thread para falar sem bloquear a interface."""
    def __init__(self, text, lang='pt'):
        super().__init__()
        self.text = text
        self.lang = lang
        self.daemon = True
    
    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)  # velocidade
            engine.say(self.text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠ Erro ao falar: {e}")


def speak(text, async_mode=True):
    """Fala um texto em português."""
    if async_mode:
        thread = SpeakThread(text, 'pt')
        thread.start()
    else:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠ Erro ao falar: {e}")


def calculate_distance(box_area, image_area, base_distance=2.0):
    """Estima distância baseada no tamanho do box."""
    area_ratio = box_area / image_area
    if area_ratio < 0.001:
        return "Muito distante"
    elif area_ratio < 0.01:
        return "Distante"
    elif area_ratio < 0.05:
        return "Próximo"
    else:
        return "Muito próximo"


def analyze_image(image_path, model_path='models/yolov8n.pt', output_dir='output', speak_enabled=False):
    """Detecta, desenha e fala os objetos."""
    
    print(f"🤖 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"📷 Processando: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Erro ao carregar imagem: {image_path}")
        return
    
    h, w = img.shape[:2]
    image_area = h * w
    
    # Detecção
    results = model(img)
    r = results[0]
    
    detections = []
    detected_names = []
    
    if len(r.boxes) > 0:
        print(f"\n✅ {len(r.boxes)} objeto(s) detectado(s):\n")
        
        for idx, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            nome = model.names[cls]
            
            box_area = (x2 - x1) * (y2 - y1)
            distance = calculate_distance(box_area, image_area)
            
            detections.append({
                'nome': nome,
                'confianca': conf,
                'distancia': distance
            })
            
            detected_names.append(nome)
            
            print(f"  [{idx+1}] {nome.upper()}")
            print(f"       Confiança: {conf*100:.1f}%")
            print(f"       Distância: {distance}")
            print()
            
            # Desenhar bbox
            color = (0, 255, 0)  # verde
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label com fundo
            label = f"{nome} {conf*100:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            cv2.rectangle(img,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1),
                         color, -1)
            
            cv2.putText(img, label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Distância
            cv2.putText(img, distance,
                       (x1 + 5, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        print("⊘ Nenhum objeto detectado")
    
    # Salvar imagem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = Path(image_path).stem
    output_file = output_path / f"{filename}_detected.jpg"
    
    cv2.imwrite(str(output_file), img)
    print(f"✅ Imagem salva: {output_file}")
    
    # Falar detecções
    if speak_enabled and detected_names:
        speech_text = "Detectei "
        for i, nome in enumerate(detected_names):
            if i == len(detected_names) - 1 and len(detected_names) > 1:
                speech_text += f"e {nome}"
            elif i > 0:
                speech_text += f", {nome}"
            else:
                speech_text += nome
        speech_text += "."
        
        print(f"\n🔊 Falando: '{speech_text}'")
        speak(speech_text, async_mode=True)
    
    return detections


def analyze_file_or_camera(source, model_path='models/yolov8n.pt', output_dir='output', speak_enabled=False):
    """Analisa arquivo, URL ou câmera com áudio."""
    
    if source.lower() in ('webcam', '0', 'camera'):
        print("📷 Abrindo câmera com detecção de objetos...")
        print("Pressione 'q' para sair\n")
        
        cap = cv2.VideoCapture(0)
        model = YOLO(model_path)
        
        frame_count = 0
        last_speech_frame = -30  # falar a cada 30 frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecção
            results = model(frame)
            r = results[0]
            
            h, w = frame.shape[:2]
            image_area = h * w
            
            frame_objects = []
            
            if len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    nome = model.names[cls]
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    distance = calculate_distance(box_area, image_area)
                    
                    frame_objects.append(nome)
                    
                    # Desenhar bbox
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{nome} {conf*100:.0f}%"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    cv2.rectangle(frame,
                                 (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0] + 10, y1),
                                 color, -1)
                    
                    cv2.putText(frame, label,
                               (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    cv2.putText(frame, distance,
                               (x1 + 5, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Falar a cada 30 frames
                if speak_enabled and frame_count - last_speech_frame >= 30:
                    unique_objects = list(set(frame_objects))
                    speech_text = "Detectei "
                    for i, nome in enumerate(unique_objects):
                        if i == len(unique_objects) - 1 and len(unique_objects) > 1:
                            speech_text += f"e {nome}"
                        elif i > 0:
                            speech_text += f", {nome}"
                        else:
                            speech_text += nome
                    speech_text += "."
                    
                    speak(speech_text, async_mode=True)
                    last_speech_frame = frame_count
            
            # Info na tela
            info_text = f"Frame: {frame_count} | Objetos: {len(frame_objects)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Detecção de Objetos 🎥', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif source.startswith('http://') or source.startswith('https://'):
        print(f"🌐 Baixando imagem: {source}")
        resp = requests.get(source, timeout=10)
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        temp_path = Path('/tmp/temp_image.jpg')
        cv2.imwrite(str(temp_path), img)
        
        analyze_image(temp_path, model_path, output_dir, speak_enabled)
    
    else:
        # Arquivo local
        analyze_image(source, model_path, output_dir, speak_enabled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detectar objetos e falar nomes')
    parser.add_argument('source', default='webcam', nargs='?',
                       help='Arquivo de imagem, URL ou "webcam"')
    parser.add_argument('--model', default='models/yolov8n.pt',
                       help='Caminho para o modelo (.pt)')
    parser.add_argument('--output', default='output',
                       help='Diretório de saída')
    parser.add_argument('--speak', action='store_true',
                       help='Ativar síntese de voz (TTS)')
    
    args = parser.parse_args()
    
    analyze_file_or_camera(args.source, args.model, args.output, args.speak)
