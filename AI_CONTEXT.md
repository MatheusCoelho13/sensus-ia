# Contexto da IA do Projeto

Este arquivo resume as informações úteis para a parte de IA/visão computacional deste projeto.

## Visão geral

- Projeto: API FastAPI com YOLOv8 para detecção de objetos e geração de orientação para navegação assistiva.
- Entrada principal: imagem enviada por upload em `/analisar` ou URL em `/analisar_url`.
- Saída principal: lista de objetos detectados, posição relativa, estimativa de proximidade e texto de orientação.
- O modelo é carregado em `app.py` e pode ser trocado por variável de ambiente `IA_MODEL_PATH`.

## Fluxo de inferência

1. A imagem é carregada e decodificada com OpenCV.
2. O modelo YOLO roda com `conf=0.20` e `iou=0.5`.
3. Cada detecção vira um objeto com:
   - nome da classe
   - lado na imagem: esquerda, centro ou direita
   - proximidade: próximo ou distante
   - bbox: coordenadas `[x1, y1, x2, y2]`
   - confidence
   - distance_cm quando for possível estimar
4. O sistema monta uma orientação curta com base em objetos centrais e obstáculos laterais.

## Endpoints relevantes

- `GET /health`: status da API.
- `GET /model`: informa se o modelo está carregado, qual caminho foi usado e quais classes estão ativas.
- `POST /model`: troca o modelo carregado via JSON com `path`.
- `POST /analisar`: analisa imagem enviada como arquivo.
- `POST /analisar_url`: analisa imagem baixada a partir de uma URL pública.

## Classes e dataset

As classes disponíveis dependem do modelo carregado. No repositório existem dois conjuntos principais:

- COCO filtrado em `config/data.yaml` com 10 classes:
  - person, backpack, chair, bench, laptop, cell phone, bottle, book, cup, tv
- Custom em `config/data_custom.yaml` com 5 classes:
  - person, backpack, chair, bench, laptop

A aplicação usa `model.names`, então o conjunto real de classes pode mudar conforme o peso carregado.

## Regras úteis para a IA

- Objeto no centro da imagem tem prioridade maior do que objeto nas bordas.
- Pessoa deve ter prioridade sobre outros objetos.
- Se houver obstáculo no centro, a orientação tenta sugerir um lado livre.
- Objeto com bbox maior tende a ser interpretado como mais próximo.
- `distance_cm` é estimativa, não medição real.

## Problemas comuns que a IA deve considerar

### Delay

- Delay é a latência entre capturar a imagem e obter a resposta.
- Pode vir de:
  - modelo pesado demais
  - execução em CPU em vez de GPU
  - imagem grande demais
  - rede lenta ao usar `/analisar_url`
- Sintoma: a resposta chega atrasada e parece que a detecção não acompanha o cenário em tempo real.

### Hitbox errada

- Hitbox aqui significa bbox da detecção.
- Pode acontecer de a bbox ficar:
  - pequena demais
  - grande demais
  - deslocada para o lado errado
  - cortando parte do objeto
- Isso afeta diretamente:
  - posição esquerda/centro/direita
  - cálculo de proximidade
  - estimativa de distância

### Falso positivo

- O modelo detecta algo que não existe na cena real.
- Pode acontecer por:
  - baixa confiança do modelo
  - confusão com fundo, sombras ou texturas
  - classes parecidas
  - dataset pequeno ou mal anotado
- Exemplo: detectar cadeira onde só existe uma forma parecida.

### Detecção errada

- O objeto existe, mas o nome da classe sai errado.
- Exemplo: cadeira detectada como mesa, ou pessoa detectada como mochila.
- Causa comum:
  - poucas imagens por classe
  - labels ruins
  - classes visualmente próximas
  - modelo geral demais para o ambiente real

### Objeto “bate com outro” ou aparece “junto”

- Quando dois objetos ficam muito próximos ou sobrepostos, o modelo pode juntar ambos em uma bbox só ou inverter a classe.
- Isso é comum em cenários com:
  - oclusão parcial
  - pessoas encostadas em móveis
  - objetos pequenos perto de objetos grandes
- A IA deve tratar isso como limitação de detecção, não como falha lógica do código.

## Como interpretar a saída

- `objetos`: lista de detecções individuais.
- `orientacao`: resumo textual para navegação.
- `timestamp`: tempo da resposta.

Exemplo de leitura prática:

- Se a bbox ocupa boa parte da altura da imagem, o objeto provavelmente está mais perto.
- Se a bbox aparece no centro, ele importa mais para a navegação.
- Se a confiança estiver baixa, a detecção pode ser suspeita.

## O que a IA deve responder ao analisar um problema

Quando o usuário relatar erro, a IA deve tentar classificar em uma destas categorias:

- delay de inferência
- bbox/hitbox errada
- falso positivo
- classe errada
- objeto sobreposto ou ocluído
- problema de dataset/anotação

## Sugestão de prompt interno para a IA

Use este contexto como base para explicar falhas de visão computacional com linguagem simples:

- Identifique se o problema é de tempo, posição, classe ou confiança.
- Compare o objeto real com o nome retornado pelo modelo.
- Verifique se a bbox está coerente com o tamanho e a posição do objeto.
- Considere sobreposição entre objetos antes de concluir que o modelo errou.
- Aponte se o erro parece vir do modelo, do dataset ou do threshold de inferência.

## Observação importante

Esta IA não mede distância real com precisão. A distância em `distance_cm` é apenas uma estimativa baseada na altura da bbox e, se fornecido, no foco da câmera.
