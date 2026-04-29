# 🤖 Assistiva IA — Sistema de Visão Assistiva com YOLOv8

> API FastAPI + Model Training para detecção de objetos e geração de orientação para navegação assistida.

## 📂 Estrutura de Pastas

```
ia/
├── app.py                    👈 API principal (FastAPI)
├── Dockerfile               👈 Container
├── Makefile                 👈 Comandos rápidos ⭐
│
├── config/                  👈 Configurações
│   ├── requirements.txt     - Dependências Python
│   └── data.yaml            - Config dataset (classes, paths)
│
├── scripts/                 👈 Scripts de automação
│   ├── full_pipeline.sh     - Orquestrador completo
│   ├── fetch_wikimedia_urls.py  - Buscar URLs
│   ├── download_images.py   - Baixar com retry/delay
│   ├── prepare_dataset.py   - Limpar + gerar labels
│   ├── train.py             - Treinar YOLOv8
│   └── evaluate.py          - Avaliar modelo
│
├── data/urls/               👈 URLs para download
├── datasets/                👈 Imagens + Labels (YOLO format)
├── models/                  👈 Modelos pré-treinados
├── logs/                    👈 Histórico de execução
└── runs/                    👈 Resultados de treino
```

## 🚀 Começar em 3 Passos

### 1️⃣ Setup Inicial

```bash
source .venv/bin/activate
pip install -r config/requirements.txt
```

### 2️⃣ Rodar Pipeline Completo (Recomendado)

```bash
make pipeline          # Monitorar em tempo real
# ou
make pipeline-bg       # Em background (continua se hibernar)
```

**O que o pipeline faz:**

- ✅ Busca URLs de imagens no Wikimedia
- ✅ Baixa imagens com retry/delay automático
- ✅ Limpa imagens corrompidas
- ✅ Gera labels automáticas com YOLOv8
- ✅ Treina modelo com 50 épocas
- ✅ Avalia modelo com métricas

### 3️⃣ Usar Modelo Treinado na API

```bash
# Editar app.py para usar modelo treinado (already updated)
make api               # Rodar em http://localhost:8000
```

Para iniciar manualmente sem Makefile:

```bash
. .venv/bin/activate
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## 📋 Todos os Comandos (Makefile)

```bash
make help              # Mostrar todos os comandos
make setup             # Criar venv + instalar dependências
make install           # Apenas instalar dependências
make fetch             # Buscar URLs do Wikimedia
make download          # Baixar imagens com delay/retry
make prepare           # Limpar + gerar labels
make train             # Treinar modelo
make eval              # Avaliar modelo
make api               # Iniciar API FastAPI
make start_api         # Alias para make api
make pipeline          # Pipeline completo (tempo real)
make pipeline-bg       # Pipeline em background
make clean             # Limpar logs/cache
make distclean         # Remover tudo (inclusive datasets)
```

---

## 📖 Documentação Completa

Passos rápidos para treinar um modelo para visão assistiva:

1. Preparar dataset
   - Estrutura esperada (YOLO):
     - `datasets/images/train/*.jpg`
     - `datasets/labels/train/*.txt` (cada `.txt` com linhas: `class x_center y_center width height` normalizados)
     - `datasets/images/val/...` e `datasets/labels/val/...`

2. Configurar `data.yaml`
   - Editar `data.yaml` neste diretório para apontar `train`, `val`, `nc` e `names`.

3. Instalar dependências

```bash
python -m pip install -r requirements.txt
```

4. Treinar

```bash
python train.py
```

5. Resultados
   - Pesos gerados em `runs/train/custom_run/weights/` (por padrão)
   - Use o endpoint em `app.py` para inferência trocando o caminho do modelo para o novo peso:
     - Ex.: `model = YOLO('runs/train/custom_run/weights/best.pt')`

6. Inferência por URL
   - Endpoint: `POST /analisar_url` com JSON `{ "url": "https://.../img.jpg" }`
   - Exemplo (curl):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"url":"https://example.com/image.jpg"}' \\
  http://localhost:8000/analisar_url
```

7. Baixar imagens por URL (montar dataset)
   - Script: `download_images.py`
   - Uso:

```bash
python download_images.py urls.txt datasets --max 500 --split 0.8
```

- `urls.txt` deve conter uma URL por linha.

8. Uso com Docker
   - Buildar a imagem:

```bash
docker build -t assistiva-ia .
```

- Rodar o container (porta 8000):

```bash
docker run --rm -p 8000:8000 assistiva-ia
```

Breve explicação de ML (como funciona o treino)

- Você fornece imagens e anotações (labels) no formato YOLO: cada imagem tem um arquivo `.txt` com linhas `class x_center y_center width height` (valores normalizados entre 0 e 1).
- O script `train.py` usa um modelo pré-treinado (`yolov8n.pt`) e ajusta (fine-tune) os pesos com suas imagens.
- Durante o treino o modelo aprende a mapear pixels a caixas e classes; quanto mais dados variados e anotados corretamente, melhor a generalização.
- Após treinar, use os pesos (`best.pt`) em `app.py` para inferência em tempo real ou por URL.

Compatibilidade com modelos

- Framework: o projeto usa a biblioteca `ultralytics` (YOLOv8). O arquivo `app.py` carrega o modelo com `YOLO('<caminho_do_peso>.pt')`.
- Pesos pré-treinados: você pode usar os pesos oficiais (`yolov8n.pt`, `yolov8s.pt`, etc.) ou pesos gerados pelo treino personalizado (`runs/train/.../weights/best.pt`).
- Como trocar para seu modelo treinado: edite a linha em `app.py` onde o modelo é carregado. Exemplo:

```
model = YOLO('runs/train/custom_run/weights/best.pt')
```

- GPU/CUDA: se o host tiver CUDA disponível e o Ultralitycs detectá-la, o modelo usará GPU automaticamente. Para executar em CPU force com variáveis/flags do ultralytics conforme a documentação.
- Performance: para imagens muito grandes, redimensione antes da inferência (o `train.py` usa `imgsz=640` por padrão). Isso reduz memória e acelera inferência.

Validação rápida após trocar pesos

1. Inicie a API (`uvicorn app:app --host 0.0.0.0 --port 8000`).
2. Teste upload:

```bash
curl -F "file=@minha_imagem.jpg" http://localhost:8000/analisar
```

3. Teste por URL:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"url":"https://example.com/img.jpg"}' http://localhost:8000/analisar_url
```

Anotações e próximas etapas sugeridas:

- Se quiser detectar profundidade/obstáculos com maior precisão, combine com sensores de profundidade ou treine com imagens anotadas com distância.
- Posso adicionar scripts de avaliação, conversão para ONNX e um cliente de upload para testar com webcam/telefone.

Precisão, erro e avaliação (confirmando: é ML)

- Este projeto usa aprendizado de máquina (YOLOv8) para detectar objetos. Os resultados são probabilísticos — o modelo retorna detecções com confiança interna (score) e, após avaliação, métricas como precisão, recall e mAP indicam qualidade.
- A precisão final depende diretamente da qualidade e quantidade de dados anotados. Erros comuns: falsos positivos (detectar algo que não é) e falsos negativos (não detectar algo presente).
- Para avaliar seu modelo, use o script `evaluate.py`. Exemplo:

```bash
python evaluate.py --weights runs/train/custom_run/weights/best.pt --data data.yaml
```

Isso executa validação no conjunto `val` definido em `data.yaml` e imprime métricas (mAP, precision, recall).

Automação para coletar imagens de classes específicas e enviar à API

- Use `collect_and_upload.py` para baixar imagens listadas em `urls.txt` e enviar cada imagem ao endpoint `POST /analisar`.
- Formatos aceitáveis em `urls.txt`:
  - `https://.../img.jpg` (será salva em `collected/common/`)
  - `chair https://.../img.jpg` (será salva em `collected/chair/`)
- Exemplo de uso:

```bash
python collect_and_upload.py urls.txt --out collected --api http://localhost:8000 --save-results results.jsonl
```

O arquivo `results.jsonl` conterá uma linha JSON por imagem com a resposta do endpoint para análise e eventuais erros.

Próximos passos recomendados:

- Reunir um conjunto representativo de imagens para `chair` e `table` (pelo menos algumas centenas cada) e anotar em formato YOLO.
- Treinar com `train.py` e validar com `evaluate.py`, ajustando hiperparâmetros conforme a métrica.
- Se quiser, eu posso gerar um exemplo `urls.txt` com alguns links públicos (você confirma que quer que eu busque e inclua links públicos?).

---

## 🔍 Monitorar Treino

Se executar o pipeline em background, monitore os logs:

```bash
# Em tempo real
tail -f logs/pipeline.log
tail -f logs/train.log

# Ver métricas finais
cat runs/detect/custom_run/results.csv
```

## 🛠️ Troubleshooting

**Problema:** Hibernação interrompe o treinamento

- **Solução:** Use `make pipeline-bg` que roda em background com `nohup`

**Problema:** Imagens corrompidas no download

- **Solução:** `prepare_dataset.py` remove automaticamente

**Problema:** Modelo não aprende (métricas zeradas)

- **Solução:** Sem labels = sem aprendizado. Execute `make prepare` para gerar labels automáticas

**Problema:** Rate limit (HTTP 429) no Wikimedia

- **Solução:** `make download` já tem delays automáticos (`--delay 2.0 --wikimedia-delay 3.0`)

## 📊 Resultados Esperados

Após treinar:

- Pasta `runs/detect/custom_run/weights/` com `best.pt` e `last.pt`
- Arquivo `runs/detect/custom_run/results.csv` com métricas por época
- mAP50, Precision, Recall e outras métricas em `make eval`

## 🐳 Docker

```bash
docker build -t assistiva-ia .
docker run --rm -p 8000:8000 assistiva-ia
```
