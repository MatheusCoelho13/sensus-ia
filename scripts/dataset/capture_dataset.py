#!/usr/bin/env python3
"""Capture frames from webcam and auto-annotate using YOLO model.

Saves images and YOLO-format label files under a dataset folder, split by view
(`front` or `side`). Press SPACE to save a frame, V to toggle view, Q to quit.

Example:
  python3 capture_dataset.py --out ../datasets/captured --model runs/detect/custom_run/weights/best.pt --conf 0.2
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def ensure_dirs(base):
    (base / 'images' / 'front').mkdir(parents=True, exist_ok=True)
    (base / 'images' / 'side').mkdir(parents=True, exist_ok=True)
    (base / 'labels' / 'front').mkdir(parents=True, exist_ok=True)
    (base / 'labels' / 'side').mkdir(parents=True, exist_ok=True)


def yolo_write_label(path: Path, boxes, classes, img_w, img_h):
    # boxes: list of [x1,y1,x2,y2], classes: list of int
    lines = []
    for (x1, y1, x2, y2), cls in zip(boxes, classes):
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        # normalize
        lines.append(f"{int(cls)} {xc/img_w:.6f} {yc/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}")
    path.write_text("\n".join(lines))


def run_capture(out_dir: Path, model_path: str, conf: float, cam_index: int, imgsz: int):
    out_dir = out_dir.resolve()
    ensure_dirs(out_dir)

    # load model
    if Path(model_path).exists():
        model = YOLO(model_path)
    else:
        print(f"Modelo não encontrado em {model_path}, tentando pretreinado...")
        model = YOLO('models/yolov8n.pt')

    names = getattr(model, 'names', None)
    print('Classes do modelo:', names)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print('Não foi possível abrir a câmera', cam_index)
        return

    view = 'front'
    counter = 0
    # find starting index
    existing = list((out_dir / 'images' / view).glob('*.jpg'))
    if existing:
        counter = max(int(p.stem.split('_')[-1]) for p in existing) + 1

    print('Controles: SPACE = salvar frame, V = alternar view (front/side), Q = sair')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Falha ao ler frame')
            break

        # run quick inference for visual feedback
        results = model(frame, conf=conf, imgsz=imgsz)
        # draw boxes on preview
        preview = frame.copy()
        boxes = []
        classes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                confv = float(box.conf[0]) if hasattr(box, 'conf') else None
                boxes.append((x1, y1, x2, y2))
                classes.append(cls)
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                label = f"{names[cls] if names else cls} {confv:.2f}" if confv is not None else f"{names[cls] if names else cls}"
                cv2.putText(preview, label, (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(preview, f'VIEW: {view}  | Press SPACE to save', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('capture', preview)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('v'):
            view = 'side' if view == 'front' else 'front'
            print('View:', view)
        elif k == 32:  # SPACE
            # save image and label
            fname = f'{view}_{int(time.time())}_{counter:04d}.jpg'
            img_path = out_dir / 'images' / view / fname
            cv2.imwrite(str(img_path), frame)
            # write labels
            label_path = out_dir / 'labels' / view / (img_path.stem + '.txt')
            if boxes:
                yolo_write_label(label_path, boxes, classes, frame.shape[1], frame.shape[0])
            else:
                label_path.write_text('')
            print(f'Salvo {img_path} (+{len(boxes)} boxes)')
            counter += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='../datasets/captured', help='Pasta base para salvar imagens/labels')
    parser.add_argument('--model', default='runs/detect/custom_run/weights/best.pt', help='Caminho do modelo YOLO (.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confiança mínima para deteções')
    parser.add_argument('--cam', type=int, default=0, help='Índice da câmera (default 0)')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamanho (imgsz) para inferência rápida')
    args = parser.parse_args()

    out_dir = Path(args.out)
    run_capture(out_dir, args.model, args.conf, args.cam, args.imgsz)


if __name__ == '__main__':
    main()
