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

# Copiar apenas o necessário para executar a API e scripts de treino.
# Datasets, runs e models não são copiados (use bind-mount em `docker run`).
COPY app.py /app/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY static/ /app/static/

# Criar diretórios de destino (se necessários)
RUN mkdir -p /app/models /app/runs /app/datasets

# Expor porta da API
EXPOSE 8000

# Entrada padrão (para API). Use o Makefile/`docker run` para rodar comandos de treino.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]