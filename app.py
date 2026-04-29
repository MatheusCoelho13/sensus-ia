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
from collections import defaultdict, deque

import time as _time

app = FastAPI()

INFERENCE_CONF = float(os.environ.get("IA_INFERENCE_CONF", "0.25"))
INFERENCE_IOU = float(os.environ.get("IA_INFERENCE_IOU", "0.45"))
INFERENCE_IMGSZ = int(os.environ.get("IA_INFERENCE_IMGSZ", "640"))
MIN_BOX_AREA_RATIO = float(os.environ.get("IA_MIN_BOX_AREA_RATIO", "0.0015"))
MAX_BOX_AREA_RATIO = float(os.environ.get("IA_MAX_BOX_AREA_RATIO", "0.85"))
MIN_BOX_ASPECT_RATIO = float(os.environ.get("IA_MIN_BOX_ASPECT_RATIO", "0.12"))
MAX_BOX_ASPECT_RATIO = float(os.environ.get("IA_MAX_BOX_ASPECT_RATIO", "8.0"))
MIN_PERSISTENCE = int(os.environ.get("IA_MIN_PERSISTENCE", "1"))
CAMERA_HISTORY_SIZE = int(os.environ.get("IA_CAMERA_HISTORY_SIZE", "5"))
ENABLE_TRACKING = os.environ.get("IA_ENABLE_TRACKING", "1").strip().lower() not in {"0", "false", "no"}

_history = defaultdict(lambda: deque(maxlen=CAMERA_HISTORY_SIZE))

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
          
            pretrained = "models/yolov8n-seg.pt"
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

CLASS_MIN_CONF = {
    "person": 0.30,
    "pessoa": 0.30,
    "chair": 0.28,
    "cadeira": 0.28,
    "backpack": 0.30,
    "laptop": 0.35,
    "bench": 0.28,
    "table": 0.28,
    "mesa": 0.28,
}

CLASS_PRIORITY = {
    "person": 3.0,
    "pessoa": 3.0,
    "chair": 1.5,
    "cadeira": 1.5,
    "table": 1.5,
    "mesa": 1.5,
    "laptop": 1.2,
    "backpack": 1.0,
    "bench": 1.2,
}


def _supports_half_precision() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """Reduz custo quando a imagem é muito grande sem forçar perda agressiva de qualidade."""
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side <= INFERENCE_IMGSZ:
        return img

    scale = INFERENCE_IMGSZ / float(max_side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _load_image_from_bytes(contents: bytes) -> Optional[np.ndarray]:
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return _prepare_image(img)


def _box_is_plausible(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> bool:
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return False

    area_ratio = (box_w * box_h) / float(w * h)
    if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
        return False

    aspect_ratio = box_w / max(1.0, box_h)
    if aspect_ratio < MIN_BOX_ASPECT_RATIO or aspect_ratio > MAX_BOX_ASPECT_RATIO:
        return False

    return True


def _run_inference(img: np.ndarray):
    model = get_model()
    return model.predict(
        img,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        imgsz=INFERENCE_IMGSZ,
        half=_supports_half_precision(),
        verbose=False,
    )


def _run_tracked_inference(img: np.ndarray):
    model = get_model()
    return model.track(
        img,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        imgsz=INFERENCE_IMGSZ,
        half=_supports_half_precision(),
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )


def _coalesce_history(camera_id: Optional[str], objetos: list[dict]) -> list[dict]:
    if not camera_id:
        return objetos

    history = _history[camera_id]
    history.append(objetos)
    if len(history) < MIN_PERSISTENCE:
        return objetos

    counts = defaultdict(int)
    confidence_sum = defaultdict(float)
    latest = {}
    for frame in history:
        for obj in frame:
            key = (
                obj.get("track_id") if obj.get("track_id") is not None else obj.get("nome"),
                obj.get("lado"),
            )
            counts[key] += 1
            confidence_sum[key] += float(obj.get("confidence") or 0.0)
            latest[key] = obj

    filtered = []
    for key, obj in latest.items():
        if counts[key] >= MIN_PERSISTENCE:
            avg_conf = confidence_sum[key] / float(counts[key])
            if avg_conf:
                obj = {**obj, "confidence": round(avg_conf, 4)}
            filtered.append(obj)
    return filtered


def _build_orientation(objetos: list[dict]) -> str:
    sides = {"esquerda": [], "centro": [], "direita": []}
    obstacle_scores = {"esquerda": 0.0, "centro": 0.0, "direita": 0.0}

    for o in objetos:
        lado = o.get("lado", "centro")
        nome = str(o.get("nome", "")).lower()
        conf = float(o.get("confidence") or 0.0)
        is_close = bool(o.get("isClose"))
        priority = CLASS_PRIORITY.get(nome, 0.0)
        sides.setdefault(lado, []).append(nome)

        if nome in {"person", "pessoa"}:
            obstacle_scores[lado] += 2.5 + priority + conf
        elif nome in OBSTACLE_CLASSES or is_close:
            obstacle_scores[lado] += 1.0 + priority + conf

    guidance_parts = []
    for lado in ["centro", "esquerda", "direita"]:
        if any(n in {"person", "pessoa"} for n in sides.get(lado, [])):
            guidance_parts.append("Pessoa à frente" if lado == "centro" else f"Pessoa à {lado}")

    if obstacle_scores["centro"] > 0:
        free_side = None
        if obstacle_scores["esquerda"] == 0:
            free_side = "esquerda"
        elif obstacle_scores["direita"] == 0:
            free_side = "direita"

        if free_side:
            guidance_parts.append(f"Obstáculo à frente — siga para a {free_side}")
        else:
            guidance_parts.append("Obstáculo à frente — cuidado, espaço estreito")
    else:
        guidance_parts.append("Caminho livre à frente")

    for lado in ["esquerda", "direita"]:
        if sides.get(lado):
            guidance_parts.append(f"{sides[lado][0]} à {lado}")

    return ", ".join(guidance_parts)


def _process_image(img: np.ndarray, focal_length_px: Optional[float], real_height_cm: Optional[float], camera_id: Optional[str] = None):
    h, w = img.shape[:2]
    use_tracking = ENABLE_TRACKING and camera_id is not None
    results = _run_tracked_inference(img) if use_tracking else _run_inference(img)

    objetos = []
    model = get_model()

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else None

            cls = int(box.cls[0])
            nome = str(model.names[cls])
            min_conf = CLASS_MIN_CONF.get(nome.lower(), INFERENCE_CONF)
            if conf is not None and conf < min_conf:
                continue

            track_id = None
            try:
                if getattr(box, "id", None) is not None:
                    track_id = int(box.id[0])
            except Exception:
                track_id = None

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if not _box_is_plausible(x1, y1, x2, y2, w, h):
                continue

            cx = (x1 + x2) / 2
            bbox_h = y2 - y1
            height_ratio = bbox_h / float(h) if h else 0.0

            if cx < w / 3:
                lado = "esquerda"
            elif cx > 2 * w / 3:
                lado = "direita"
            else:
                lado = "centro"

            proximidade = "próximo" if height_ratio > PROXIMITY_HEIGHT_THRESHOLD else "distante"

            distance_cm = None
            try:
                if focal_length_px is not None:
                    rh = real_height_cm if real_height_cm is not None else CLASS_REAL_HEIGHT_CM.get(nome.lower())
                    if rh is not None and bbox_h > 0:
                        distance_cm = round(float(focal_length_px * rh / bbox_h), 1)
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
                "track_id": track_id,
            })

    objetos = _coalesce_history(camera_id, objetos)
    orientacao = _build_orientation([
        {**o, "isClose": o.get("proximidade") == "próximo"}
        for o in objetos
    ])

    objetos_contrato = []
    for o in objetos:
        prox = o.get("proximidade", "distante")
        objetos_contrato.append({
            "nome": o["nome"],
            "distancia": "perto" if prox == "próximo" else "longe",
            "isClose": prox == "próximo",
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


@app.post("/analisar")
async def analisar(file: UploadFile, focal_length_px: Optional[float] = None, real_height_cm: Optional[float] = None, camera_id: Optional[str] = None):
    contents = await file.read()

    img = _load_image_from_bytes(contents)

    if img is None:
        return {"error": "imagem inválida"}
    return _process_image(img, focal_length_px, real_height_cm, camera_id=camera_id)


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
async def analisar_url(item: URLItem, focal_length_px: Optional[float] = None, real_height_cm: Optional[float] = None, camera_id: Optional[str] = None):
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
    img = _load_image_from_bytes(contents)

    if img is None:
        raise HTTPException(status_code=400, detail="imagem inválida ou formato não suportado")
    return _process_image(img, focal_length_px, real_height_cm, camera_id=camera_id)