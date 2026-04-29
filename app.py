from fastapi import FastAPI, UploadFile, HTTPException, Header
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from typing import Optional
import ipaddress
import httpx
import os
import re
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import time as _time

app = FastAPI()

# servir página estática para teste da câmera
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health():
    return {"status": "UP"}


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get('/model')
async def model_info():
    """Retorna caminho do modelo atual (se carregado) e se está carregado."""
    loaded = _model is not None
    path = globals().get('_model_path', None)
    device = None
    classes = None
    try:
        if loaded:
            # classes/names
            names = getattr(_model, 'names', None)
            if isinstance(names, dict):
                # sort by key to preserve class indices
                try:
                    classes = [names[k] for k in sorted(names, key=lambda x: int(x))]
                except Exception:
                    classes = [names[k] for k in names]
            else:
                classes = names

            # try to detect device (works if model is a torch module)
            try:
                import torch
                md = getattr(_model, 'model', None)
                dev = None
                if md is not None:
                    # check first parameter
                    for p in md.parameters():
                        dev = p.device
                        break
                if dev is None:
                    dev = getattr(_model, 'device', None) or getattr(md, 'device', None)
                device = str(dev) if dev is not None else None
            except Exception:
                # fallback
                device = str(getattr(_model, 'device', None) or getattr(getattr(_model, 'model', None), 'device', None))
    except Exception:
        pass

    return {"loaded": loaded, "model_path": path, "device": device, "classes": classes}


_API_KEY = os.environ.get("IA_API_KEY", "").strip()


def _verificar_api_key(x_api_key: Optional[str]) -> None:
    """Se IA_API_KEY estiver definida, exige o header X-Api-Key correspondente."""
    if not _API_KEY:
        return
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Não autorizado")


@app.post('/model')
async def model_set(item: dict, x_api_key: Optional[str] = Header(default=None)):
    """Define um novo modelo via JSON {"path":"..."}. Requer X-Api-Key se IA_API_KEY estiver configurada."""
    _verificar_api_key(x_api_key)
    path = item.get('path') if isinstance(item, dict) else None
    if not path:
        raise HTTPException(status_code=400, detail='Forneça JSON com chave "path"')
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail='Modelo não encontrado')
    # reload model
    try:
        global _model
        global _model_path
        _model = YOLO(path)
        _model_path = path
        return {"model_path": _model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# carrega melhor modelo treinado ou fallback para pré-treinado
_model = None
_model_path = None

def get_model():
    """Carrega (e cacheia) o modelo na primeira requisição para evitar carregar no import."""
    global _model
    global _model_path
    if _model is None:
        # IA_MODEL_PATH permite trocar o modelo sem alterar o código.
        # Exemplos: IA_MODEL_PATH=models/yolov8n.pt  ou  IA_MODEL_PATH=runs/detect/meu_treino/weights/best.pt
        env_path = os.environ.get("IA_MODEL_PATH", "").strip()
        if env_path:
            chosen = env_path
            if not Path(chosen).exists():
                raise FileNotFoundError(f"Modelo definido em IA_MODEL_PATH não encontrado: {chosen}")
        else:
            trained_model = "runs/detect/custom_run/weights/best.pt"
            pretrained = "models/yolov8n.pt"
            chosen = trained_model if Path(trained_model).exists() else pretrained

        try:
            print(f"Carregando modelo: {chosen}")
            _model = YOLO(chosen)
            _model_path = chosen
        except Exception as e:
            print(f"Erro ao carregar modelo {chosen}: {e}")
            raise

    return _model

# classes que normalmente representam obstáculos móveis/estáticos
OBSTACLE_CLASSES = {"chair", "sofa", "couch", "bench", "person", "bicycle", "motorbike", "car", "bed", "dining table", "potted plant", "tv", "laptop", "backpack", "suitcase", "handbag", "bottle", "umbrella", "table"}

# altura relativa da bbox acima da qual consideramos 'próximo'
PROXIMITY_HEIGHT_THRESHOLD = 0.25

# alturas típicas em centímetros para algumas classes (usadas para estimativa de distância)
CLASS_REAL_HEIGHT_CM = {
    "person": 170.0,
    "pessoa": 170.0,
    "chair": 90.0,
    "cadeira": 90.0,
    "table": 75.0,
    "mesa": 75.0,
    "bottle": 25.0,
}


@app.post("/analisar")
async def analisar(file: UploadFile, focal_length_px: Optional[float] = None, real_height_cm: Optional[float] = None):
    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "imagem inválida"}

    h, w = img.shape[:2]
    model = get_model()
    results = model(img, conf=0.20, iou=0.5, verbose=False)

    objetos = []
    sides = {"esquerda": [], "centro": [], "direita": []}
    obstacle_counts = {"esquerda": 0, "centro": 0, "direita": 0}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            nome = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # confidence may be available as box.conf
            try:
                conf = float(box.conf[0])
            except Exception:
                conf = None
            cx = (x1 + x2) / 2
            bbox_h = (y2 - y1)
            height_ratio = bbox_h / float(h) if h else 0.0

            if cx < w / 3:
                lado = "esquerda"
            elif cx > 2 * w / 3:
                lado = "direita"
            else:
                lado = "centro"

            proximidade = "próximo" if height_ratio > PROXIMITY_HEIGHT_THRESHOLD else "distante"

            # estimativa de distância em cm (se fornecido focal_length_px ou houver tamanho conhecido da classe)
            distance_cm = None
            try:
                if focal_length_px is not None:
                    rh = real_height_cm
                    if rh is None:
                        rh = CLASS_REAL_HEIGHT_CM.get(nome.lower())
                    if rh is not None and bbox_h > 0:
                        distance_cm = float(focal_length_px * rh / bbox_h)
                        distance_cm = round(distance_cm, 1)
            except Exception:
                distance_cm = None

            objetos.append({
                "nome": nome,
                "lado": lado,
                "proximidade": proximidade,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "height_ratio": height_ratio,
                "distance_cm": distance_cm,
            })
            sides[lado].append(nome)

            if nome.lower() in OBSTACLE_CLASSES or proximidade == "próximo":
                obstacle_counts[lado] += 1

    # construir orientação concisa
    guidance_parts = []

    # Priorizar pessoas
    for lado in ["centro", "esquerda", "direita"]:
        if any(n.lower() == "person" or n.lower() == "pessoa" for n in sides[lado]):
            if lado == "centro":
                guidance_parts.append("Pessoa à frente")
            else:
                guidance_parts.append(f"Pessoa à {lado}")

    # Se há obstáculo no centro, sugerir desvio quando possível
    if obstacle_counts["centro"] > 0:
        free_side = None
        if obstacle_counts["esquerda"] == 0:
            free_side = "esquerda"
        elif obstacle_counts["direita"] == 0:
            free_side = "direita"

        if free_side:
            guidance_parts.append(f"Obstáculo à frente — siga para a {free_side}")
        else:
            guidance_parts.append("Obstáculo à frente — cuidado, espaço estreito")
    else:
        guidance_parts.append("Caminho livre à frente")

    # adicionar um item notável por lado (máx. 1)
    for lado in ["esquerda", "direita"]:
        if sides[lado]:
            guidance_parts.append(f"{sides[lado][0]} à {lado}")

    orientacao = ", ".join(guidance_parts)

    objetos_contrato = []
    for o in objetos:
        prox = o.get("proximidade", "distante")
        if prox == "próximo":
            distancia = "perto"
            is_close = True
        else:
            distancia = "longe"
            is_close = False
        objetos_contrato.append({
            "nome": o["nome"],
            "distancia": distancia,
            "isClose": is_close,
            "lado": o.get("lado"),
            "confidence": o.get("confidence"),
            "distance_cm": o.get("distance_cm"),
            "bbox": o.get("bbox"),
        })

    return {
        "objetos": objetos_contrato,
        "orientacao": orientacao,
        "timestamp": int(_time.time()),
    }


_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

def _validar_url_publica(url: str) -> None:
    """Bloqueia URLs que apontam para IPs privados/loopback (SSRF)."""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="URL deve usar http ou https")
    from urllib.parse import urlparse
    import socket
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="URL inválida")
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except Exception:
        raise HTTPException(status_code=400, detail="Não foi possível resolver o host da URL")
    if any(ip in net for net in _PRIVATE_NETS):
        raise HTTPException(status_code=400, detail="URL aponta para endereço privado — não permitido")


class URLItem(BaseModel):
    url: str


@app.post("/analisar_url")
async def analisar_url(item: URLItem, focal_length_px: Optional[float] = None, real_height_cm: Optional[float] = None):
    url = item.url
    _validar_url_publica(url)
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            contents = resp.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Não foi possível baixar a imagem")
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="imagem inválida ou formato não suportado")

    h, w = img.shape[:2]
    model = get_model()
    results = model(img, conf=0.20, iou=0.5, verbose=False)

    objetos = []
    sides = {"esquerda": [], "centro": [], "direita": []}
    obstacle_counts = {"esquerda": 0, "centro": 0, "direita": 0}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            nome = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            try:
                conf = float(box.conf[0])
            except Exception:
                conf = None
            cx = (x1 + x2) / 2
            bbox_h = (y2 - y1)
            height_ratio = bbox_h / float(h) if h else 0.0

            if cx < w / 3:
                lado = "esquerda"
            elif cx > 2 * w / 3:
                lado = "direita"
            else:
                lado = "centro"

            proximidade = "próximo" if height_ratio > PROXIMITY_HEIGHT_THRESHOLD else "distante"

            # estimativa de distância em cm (se fornecido focal_length_px ou houver tamanho conhecido da classe)
            distance_cm = None
            try:
                if focal_length_px is not None:
                    rh = real_height_cm
                    if rh is None:
                        rh = CLASS_REAL_HEIGHT_CM.get(nome.lower())
                    if rh is not None and bbox_h > 0:
                        distance_cm = float(focal_length_px * rh / bbox_h)
                        distance_cm = round(distance_cm, 1)
            except Exception:
                distance_cm = None

            objetos.append({
                "nome": nome,
                "lado": lado,
                "proximidade": proximidade,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "height_ratio": height_ratio,
                "distance_cm": distance_cm,
            })
            sides[lado].append(nome)

            if nome.lower() in OBSTACLE_CLASSES or proximidade == "próximo":
                obstacle_counts[lado] += 1

    # construir orientação concisa (mesma lógica que /analisar)
    guidance_parts = []

    for lado in ["centro", "esquerda", "direita"]:
        if any(n.lower() == "person" or n.lower() == "pessoa" for n in sides[lado]):
            if lado == "centro":
                guidance_parts.append("Pessoa à frente")
            else:
                guidance_parts.append(f"Pessoa à {lado}")

    if obstacle_counts["centro"] > 0:
        free_side = None
        if obstacle_counts["esquerda"] == 0:
            free_side = "esquerda"
        elif obstacle_counts["direita"] == 0:
            free_side = "direita"

        if free_side:
            guidance_parts.append(f"Obstáculo à frente — siga para a {free_side}")
        else:
            guidance_parts.append("Obstáculo à frente — cuidado, espaço estreito")
    else:
        guidance_parts.append("Caminho livre à frente")

    for lado in ["esquerda", "direita"]:
        if sides[lado]:
            guidance_parts.append(f"{sides[lado][0]} à {lado}")

    orientacao = ", ".join(guidance_parts)

    objetos_contrato = []
    for o in objetos:
        prox = o.get("proximidade", "distante")
        if prox == "próximo":
            distancia = "perto"
            is_close = True
        else:
            distancia = "longe"
            is_close = False
        objetos_contrato.append({
            "nome": o["nome"],
            "distancia": distancia,
            "isClose": is_close,
            "lado": o.get("lado"),
            "confidence": o.get("confidence"),
            "distance_cm": o.get("distance_cm"),
            "bbox": o.get("bbox"),
        })

    return {
        "objetos": objetos_contrato,
        "orientacao": orientacao,
        "timestamp": int(_time.time()),
    }