#!/usr/bin/env python3
"""Download YOLO base models for the Sensus IA service."""

import argparse
import os
from pathlib import Path

MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}

DEFAULT_MODEL = "yolov8n"
DEFAULT_DEST = Path(__file__).parent.parent / "models"


def download(model_name: str, dest: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        req = Path(__file__).parent.parent / "config" / "requirements.txt"
        print(
            "[download_model] Módulo 'ultralytics' não encontrado. Instale as dependências:"
        )
        print(f"  pip install -r {req}")
        print("ou:\n  pip install ultralytics")
        return

    dest.mkdir(parents=True, exist_ok=True)
    filename = MODELS[model_name]
    output_path = dest / filename

    if output_path.exists():
        print(f"[download_model] {output_path} já existe, pulando download.")
        return

    print(f"[download_model] Baixando {filename} → {output_path} ...")
    model = YOLO(filename)  # ultralytics baixa automaticamente em cache
    src = Path(filename)
    if src.exists():
        src.rename(output_path)
        print(f"[download_model] Salvo em {output_path}")
    else:
        # ultralytics pode salvar em ~/.cache/ultralytics
        import shutil
        import torch
        hub_dir = Path(torch.hub.get_dir()) / "ultralytics" / filename
        if hub_dir.exists():
            shutil.copy2(hub_dir, output_path)
            print(f"[download_model] Copiado de cache para {output_path}")
        else:
            print(f"[download_model] Modelo baixado (em cache do ultralytics). "
                  f"Rode com MODEL_PATH vazio para usar o cache automático.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa modelos YOLO base via ultralytics")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Modelo a baixar (padrão: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Pasta de destino (padrão: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(args.model, args.dest)


if __name__ == "__main__":
    main()
