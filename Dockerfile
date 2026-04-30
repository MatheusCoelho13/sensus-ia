FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Dependências do sistema necessárias para OpenCV, ffmpeg e utilitários
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python (requirements em config/requirements.txt)
COPY config/requirements.txt /tmp/requirements.txt
# Use BuildKit cache for pip to speed up repeated builds. Requires DOCKER_BUILDKIT=1.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && pip install -r /tmp/requirements.txt --no-cache-dir

# Copiar código e configurações
COPY app.py /app/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY static/ /app/static/

# Copiar modelos treinados para dentro da imagem (necessário em produção sem volume mount)
COPY models/ /app/models/
COPY runs/detect/custom_run/weights/best.pt /app/runs/detect/custom_run/weights/best.pt

RUN mkdir -p /app/datasets

# Expor porta da API
EXPOSE 8000

# Entrada padrão (para API). Use o Makefile/`docker run` para rodar comandos de treino.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]