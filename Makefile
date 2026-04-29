make_train:
	@echo "Buildando e rodando Dockerfile.train para treino automático"
	@DOCKER_BUILDKIT=1 docker build -f Dockerfile.train -t assistiva-ia-train .
	@docker run --rm -v $(PWD):/app -w /app assistiva-ia-train
.PHONY: help setup install prepare start api start_api start_train start_train_local start_tarin_local docker-build docker-run-dev clean clean_dataset distclean

help:
	@echo "Assistiva IA - comandos úteis:"
	@echo "  make setup         - criar venv e instalar dependências"
	@echo "  make install       - instalar dependências (venv ativo)"
	@echo "  make prepare       - preparar dataset (scripts/prepare_dataset.py)"
	@echo "  make start         - executar scripts/launcher.sh (geral)"
	@echo "  make api           - iniciar API FastAPI local (uvicorn)"
	@echo "  make start_train   - iniciar treinamento (dentro do Docker)"
	@echo "  make start_train_local - iniciar treinamento local (venv)"
	@echo "  make docker-build  - construir imagem Docker"
	@echo "  make docker-run-dev- executar container dev (shell)"
	@echo "  make clean_dataset - dry-run da limpeza do dataset (use APPLY=yes para aplicar)"
	@echo "  make clean         - remover logs e caches"
	@echo "  make distclean     - remover runs e models/best.pt"

setup:
	@if [ -f .venv/bin/activate ]; then \
		echo ".venv já existe — pulando criação"; \
	else \
		rm -rf .venv; \
		python3 -m venv --without-pip .venv; \
		python3 -c "from urllib.request import urlretrieve; urlretrieve('https://bootstrap.pypa.io/get-pip.py', '.venv/get-pip.py')"; \
		.venv/bin/python .venv/get-pip.py; \
		. .venv/bin/activate && python -m pip install -U pip && python -m pip install -r config/requirements.txt; \
	fi

install:
	. .venv/bin/activate && pip install -r config/requirements.txt

prepare:
	python3 scripts/prepare_dataset.py

start:
	bash scripts/launcher.sh

api:
	@if [ ! -f .venv/bin/activate ]; then \
		echo ".venv não encontrado; execute 'make setup' primeiro"; exit 1; \
	fi; \
	. .venv/bin/activate && uvicorn app:app --reload --host 127.0.0.1 --port 8000

start_api: api

install_train_docker:
	@echo "Executando treinamento dentro do Docker via scripts/launcher.sh"
	@DOCKER_BUILDKIT=1 docker build -t assistiva-ia .
	@docker run --rm -v $(PWD):/app -w /app -e IN_DOCKER=1 assistiva-ia bash -c "bash scripts/launcher.sh"
start_train:
	@echo "Executando treinamento com docker (venv). Use  quando tem  computador potente."
	. .venv/bin/activate && python3 scripts/train.py --max-epochs 50 --step 5 --target-map 0.60 --docker
start_train_local:
	@echo "Executando treinamento local (venv). Use somente para desenvolvimento rápido."
	@if [ ! -f .venv/bin/activate ]; then \
		echo ".venv não encontrado; execute 'make setup' primeiro"; exit 1; \
	fi; \
	. .venv/bin/activate && python3 scripts/train.py --max-epochs 50 --step 5 --target-map 0.60
start_train_local_full:
	@echo "Executando treinamento local (venv) usando TODO o dataset (train e val, sem step)."
	. .venv/bin/activate && python3 scripts/train.py --max-epochs 100 --target-map 0.60 --batch 16 --imgsz 640

start_train_gpu:
	@echo "Executando treinamento local usando GPU (exige CUDA) - TODAS as imagens, 150 épocas"
	@if [ ! -f .venv/bin/activate ]; then \
		echo ".venv não encontrado ou incompleto; execute 'make setup' primeiro"; exit 1; \
	fi; \
	. .venv/bin/activate && python3 scripts/train.py --require-cuda --device cuda:0 --max-epochs 150 --step 150 --target-map 0.60 --batch 16 --imgsz 640

start_train_gpu_balanced:
	@echo "🎯 Executando treinamento com balanceamento de classes (GPU, até 150 épocas)"
	@if [ ! -f .venv/bin/activate ]; then \
		echo ".venv não encontrado ou incompleto; execute 'make setup' primeiro"; exit 1; \
	fi; \
	. .venv/bin/activate && python3 scripts/calculate_class_weights.py --data config/data.yaml 2>&1 | tee class_weights.log; \
	WEIGHTS=$$(. .venv/bin/activate && python3 -c "import yaml; data=yaml.safe_load(open('config/class_weights.yaml')); print(','.join(str(data['class_weights'][i]) for i in range(10)))"); \
	echo "📊 Pesos calculados: $$WEIGHTS"; \
	. .venv/bin/activate && python3 scripts/train.py --require-cuda --device cuda:0 --max-epochs 150 --step 150 --target-map 0.60 --batch 16 --imgsz 640 --cls-weight "$$WEIGHTS" 2>&1 | tee training_balanced.log

show_runs:
	@. .venv/bin/activate && python3 scripts/show_runs.py --data config/data.yaml --device cpu

start_train_select:
	@echo "Start training (local) - escolha modelo com MODEL_CHOICE=base|best ou passe MODEL=path"
	@MODEL_CHOICE="$${MODEL_CHOICE:-base}"; \
	MODEL_ARG="$${MODEL:-yolov8n.pt}"; \
	if [ "$${MODEL_CHOICE}" = "best" ]; then \
		if [ -f models/best.pt ]; then MODEL_ARG="models/best.pt"; else echo "models/best.pt não encontrado, usando $(MODEL):-yolov8n.pt"; fi; \
	fi; \
	echo "Usando modelo: $$MODEL_ARG"; \
	. .venv/bin/activate && python3 scripts/train.py --model "$$MODEL_ARG" --max-epochs 50 --step 5 --target-map 0.60

start_train_docker_select:
	@echo "Start training (docker) - escolha modelo com MODEL_CHOICE=base|best ou passe MODEL=path"
	@MODEL_CHOICE="$${MODEL_CHOICE:-base}"; \
	MODEL_ARG="$${MODEL:-yolov8n.pt}"; \
	if [ "$${MODEL_CHOICE}" = "best" ]; then \
		if [ -f models/best.pt ]; then MODEL_ARG="/app/models/best.pt"; else echo "models/best.pt não encontrado, usando yolov8n.pt"; MODEL_ARG="yolov8n.pt"; fi; \
	fi; \
	@DOCKER_BUILDKIT=1 docker build -t assistiva-ia .; \
	docker run --rm -v $(PWD):/app -w /app assistiva-ia python3 scripts/train.py --model "$$MODEL_ARG" --max-epochs 50 --step 5 --target-map 0.60

docker-build:
	@echo "Construindo imagem Docker (usando BuildKit cache)"
	@DOCKER_BUILDKIT=1 docker build --progress=plain -t assistiva-ia .

docker-run-dev:
	docker run --rm -it -p 8000:8000 -v $(PWD):/app assistiva-ia bash

clean_dataset:
	@echo "Executando limpeza e correção do dataset"
	@if [ "$(APPLY)" = "yes" ]; then \
		. .venv/bin/activate && python3 scripts/clean_and_fix_dataset.py ; \
	else \
		. .venv/bin/activate && python3 scripts/clean_and_fix_dataset.py --dry-run ; \
	fi

sync:
	@echo "Sincronizar artifacts (models/ e runs/) para destino: ${SYNC_DEST:-backup/}"
	@mkdir -p ${SYNC_DEST:-backup/}
	@./scripts/sync_artifacts.sh ${SYNC_DEST:-backup/}

clean:
	@rm -f logs/*.log
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

model_clean: clean
	@rm -rf runs/ models/best.pt

.DEFAULT_GOAL := help

# Generic helper to run scripts in the scripts/ folder.
# Usage: `make scripts/<name>` will run `scripts/<name>.sh` if present,
# or `scripts/<name>.py` if present (uses venv python when available).
.PHONY: scripts-run setup_gpu auto_annotate enrich_dataset download_images fetch_wikimedia filter_coco convert_coco prepare_dataset clean_and_fix_dataset run_script
scripts-run:
	@echo "Run a script from scripts/: make scripts/<name>  (e.g. make scripts/setup_gpu)"


scripts/%:
	@echo "Running scripts/$*..."
	@if [ -f scripts/$*.sh ]; then \
		bash scripts/$*.sh $(ARGS); \
	elif [ -f scripts/$*.py ]; then \
		if [ -f .venv/bin/activate ]; then . .venv/bin/activate && python3 scripts/$*.py $(ARGS); else python3 scripts/$*.py $(ARGS); fi; \
	else \
		echo "No script found: scripts/$*.sh or scripts/$*.py"; exit 1; \
	fi

# Common convenience aliases
setup_gpu: scripts/setup_gpu
auto_annotate: scripts/auto_annotate
enrich_dataset: scripts/enrich_dataset
download_images: scripts/download_images
fetch_wikimedia: scripts/fetch_wikimedia_urls
filter_coco:
	@CLASSES="$${CLASSES:-person,backpack,chair,bench,laptop,cell phone,bottle,book,cup,tv}"; \
	OUT_DIR="$${OUT_DIR:-data/coco_filtered}"; \
	COCO_DIR="$${COCO_DIR:-data/coco}"; \
	if [ -f .venv/bin/activate ]; then . .venv/bin/activate && python3 scripts/filter_coco_by_classes.py --coco-dir "$$COCO_DIR" --out-dir "$$OUT_DIR" --classes "$$CLASSES" --symlink; else python3 scripts/filter_coco_by_classes.py --coco-dir "$$COCO_DIR" --out-dir "$$OUT_DIR" --classes "$$CLASSES" --symlink; fi
convert_coco: scripts/convert_coco_to_yolo
prepare_dataset: scripts/prepare_dataset
clean_and_fix_dataset: scripts/clean_and_fix_dataset
run_script: scripts-run

# Compatibilidade com erro de digitação comum.
start_tarin_local: start_train_local

