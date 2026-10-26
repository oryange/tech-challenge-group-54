# MedWatch — Monitoramento Multimodal de Pacientes

Tech Challenge Fase 4 (POS Tech IADT) — Grupo 54.

Sistema que analisa e funde dados multimodais (vídeo, áudio e texto/sinais vitais) para identificar sinais precoces de risco e alertar a equipe médica, usando modelos locais (OpenPose / YOLOv8) e serviços gerenciados Azure AI (Speech to Text e Language/Text Analytics).

- Arquitetura: [docs/architecture.md](docs/architecture.md)
- Plano de desenvolvimento e checklists de PR: [docs/plano_desenvolvimento.md](docs/plano_desenvolvimento.md)

## Status

| Módulo | Status |
|---|---|
| Setup / infraestrutura (PR 1) | ✅ |
| Core: eventos e alertas (PR 2) | ✅ |
| Sinais vitais (PR 3) | ✅ |
| Áudio + Azure (PR 4) | ⏳ |
| Vídeo: YOLOv8 + OpenPose (PR 5) | ✅ |
| Prescrições e movimentação (PR 6) | ⏳ |
| Fusão, API e Dashboard (PR 7) | ⏳ |
| Relatório técnico e demo (PR 8) | ⏳ |

## Pré-requisitos

- Python 3.11 ou 3.12
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)
- Docker (opcional)
- Azure CLI (`az`) e uma assinatura Azure (para os módulos de áudio)

## Setup local

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install          # pip install -r requirements.txt + hook do pre-commit
cp .env.example .env  # preencha as variáveis Azure
medwatch healthcheck
```

## Setup com Docker

```bash
make docker-build
docker compose run --rm medwatch healthcheck
```

## Provisionamento Azure

O projeto roda localmente; o Azure é usado apenas como API (Speech to Text e Language/Text Analytics, requisitos obrigatórios do enunciado). Nada é hospedado no Azure. O plano gratuito **F0** basta para o projeto (cerca de 5 h de áudio/mês no Speech e 5 mil registros de texto/mês no Language).

### Opção A — Portal Azure (sem instalar nada)

1. Crie uma conta em https://azure.microsoft.com/free/students (Azure for Students, sem cartão) ou use uma assinatura pessoal.
2. No portal, crie um recurso **Speech** (busque por "Speech services"):
   - Região: **Brazil South** · Tipo de preço: **Free F0**
   - Depois de criado: menu **Chaves e Ponto de Extremidade** → copie a **Chave 1** e a **Região**.
3. Crie um recurso **Language** (busque por "Language service"; pule os recursos adicionais e aceite o aviso de IA Responsável):
   - Região: **Brazil South** · Tipo de preço: **Free F0**
   - Em **Chaves e Ponto de Extremidade** → copie a **Chave 1** e o **Ponto de extremidade**.
4. Crie o `.env` e preencha:

```bash
cp .env.example .env
# AZURE_SPEECH_KEY=<chave 1 do Speech>
# AZURE_SPEECH_REGION=brazilsouth
# AZURE_LANGUAGE_ENDPOINT=<ponto de extremidade do Language>
# AZURE_LANGUAGE_KEY=<chave 1 do Language>
```

### Opção B — Azure CLI

```bash
brew install azure-cli   # ou https://learn.microsoft.com/cli/azure/install-azure-cli
az login
./scripts/provision_azure.sh   # cria RG, Speech (F0) e Language (F0) em brazilsouth
```

O script imprime as variáveis para copiar no `.env`.

### Validação

```bash
medwatch healthcheck --online
```

Deve mostrar ✅ em `azure.speech.online` e `azure.language.online`. Se aparecer ❌, o log indica a causa (chave recusada com HTTP 401, endpoint inacessível etc.).

> O `.env` está no `.gitignore`: nunca commite chaves. Limite do F0: um recurso gratuito de cada tipo por assinatura.

Para remover tudo ao final: apague o grupo de recursos no portal (ou `az group delete -n rg-medwatch-g54`).

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MEDWATCH_ENV` | `dev` | `dev`, `test` ou `prod` |
| `MEDWATCH_LOG_LEVEL` | `INFO` | Nível de log no console |
| `MEDWATCH_LOG_JSON` | `false` | Log do console em JSON |
| `MEDWATCH_AZURE_MOCK` | `false` | Usa respostas gravadas no lugar do Azure (CI/offline) |
| `MEDWATCH_POSE_BACKEND` | `yolo` | `yolo` (YOLOv8-pose) ou `openpose` |
| `MEDWATCH_DATA_DIR` / `MODELS_DIR` / `OUTPUTS_DIR` | `./data`, `./models`, `./outputs` | Diretórios de trabalho |
| `MEDWATCH_ALERT_MIN_SEVERITY` | `medium` | Severidade mínima para um evento virar alerta |
| `MEDWATCH_ALERT_COOLDOWN_MINUTES` | `15` | Janela de supressão de alertas repetidos |
| `MEDWATCH_ALERT_WEBHOOK_URL` | — | Webhook opcional para receber os alertas |
| `MEDWATCH_ALERT_WEBHOOK_FORMAT` | `generic` | `generic`, `teams`, `slack` ou `discord` |
| `AZURE_SPEECH_KEY` | — | Chave do recurso Speech |
| `AZURE_SPEECH_REGION` | `brazilsouth` | Região do recurso Speech |
| `AZURE_SPEECH_LANGUAGE` | `pt-BR` | Idioma da transcrição |
| `AZURE_LANGUAGE_ENDPOINT` | — | Endpoint do recurso Language |
| `AZURE_LANGUAGE_KEY` | — | Chave do recurso Language |

## Sinais vitais

Detecção de anomalias em sinais vitais do **MIMIC-IV Clinical Database Demo** (PhysioNet, acesso aberto, 100 pacientes), combinando três técnicas: **NEWS2** (risco clínico absoluto), **z-score robusto** (desvio em relação ao histórico do próprio paciente) e **Isolation Forest** (combinação rara de sinais na população). Método, calibração e resultados em [docs/relatorio_vitais.md](docs/relatorio_vitais.md).

```bash
medwatch data download                               # ~6 MB, confere o SHA-256 de cada arquivo
medwatch vitals list                                 # pacientes disponíveis (subject_id)
medwatch vitals analyze --patient 10004235           # eventos + outputs/reports/vitals_<id>.json e vitals_<id>_<stay>.png
medwatch vitals analyze --patient 10004235 --alert   # envia os eventos ao AlertManager
medwatch vitals evaluate                             # precision/recall/F1 com anomalias injetadas
```

Resultado da avaliação (283 anomalias sintéticas injetadas em 116 internações reais, as que têm pelo menos 24 h de dados, seed 42):

| Método | Precision | Recall | F1 |
|---|---|---|---|
| z-score robusto | 1,000 | 0,767 | 0,868 |
| Isolation Forest | 0,949 | 0,191 | 0,318 |
| NEWS2 | 0,753 | 0,187 | 0,300 |
| **Combinado** | **0,940** | **0,830** | **0,882** |

A precision é **de injeção**, não clínica: mede quantas das detecções novas, que surgiram com a injeção, coincidem com uma anomalia injetada. A precision clínica real exigiria rótulos de especialistas (ver [relatório](docs/relatorio_vitais.md#avaliação)).

Limiares clínicos e parâmetros dos detectores ficam em [configs/thresholds.yaml](configs/thresholds.yaml).

## Vídeo clínico

Análise postural com **YOLOv8-pose** (padrão) ou **OpenPose** (rede original da CMU em PyTorch), detecção de objetos e áreas críticas com **YOLOv8 + ByteTrack**, contagem de repetições, desvios de exercício, **detecção de quedas** e relatório automático. Método e resultados em [docs/relatorio_video.md](docs/relatorio_video.md).

```bash
medwatch data download --dataset videos      # vídeos de exemplo (licença aberta)
medwatch video analyze agachamento_unipodal.webm --exercise agachamento_unipodal
medwatch video analyze agachamento_barra.webm --exercise agachamento --backend openpose
medwatch video analyze urfall_queda_01.mp4 --alert          # queda → alerta CRITICAL
medwatch video analyze meu_video.mp4 --scene quarto         # áreas críticas (ativa objetos)
medwatch video analyze meu_video.mp4 --start-time 2026-10-19T14:30:00-03:00  # horário real

medwatch data download --dataset urfall      # 40 sequências do UR Fall Dataset
medwatch video evaluate-falls                # acurácia: calibração 01–10, validação 11–20
```

Exercícios disponíveis: `agachamento`, `agachamento_unipodal` e `elevacao_bracos`. Limiares, cenas e áreas críticas ficam em [configs/video_rules.yaml](configs/video_rules.yaml). Os pesos dos modelos são baixados para `models/` no primeiro uso, com SHA-256 conferido (YOLOv8n, YOLOv8n-pose e OpenPose).

Cada análise gera em `outputs/reports/video/` um relatório HTML autocontido, um JSON com os eventos, o vídeo anotado e a série de pose (`.parquet`). Sem `--start-time`, os eventos são datados a partir do momento da análise; para fundir com os vitais (PR 7), informe o início real da gravação. Se o OpenCV não tiver H.264 (caso do pip no Linux/Docker), o vídeo anotado é convertido com o `ffmpeg` para abrir no navegador.

Detecção de quedas (UR Fall Dataset, conjunto de validação 11–20, sem ajuste de limiares):

| Backend | Quedas detectadas | Alarmes falsos | Velocidade |
|---|---|---|---|
| YOLOv8-pose | 6/10 | 0/10 | 43–105 quadros/s |
| OpenPose | 4/10 | 0/10 | 2–11 quadros/s |

## Alertas

Todos os módulos publicam eventos clínicos em um barramento; o `AlertManager` os transforma em alertas para a equipe médica (detalhes em [docs/architecture.md](docs/architecture.md#contratos-de-dados-e-fluxo-de-alertas)).

```bash
medwatch alerts demo
```

Dispara um cenário fictício pelo fluxo completo e mostra os alertas no console. Resultado esperado: `Alertas enviados: 4 · suprimidos (cooldown): 1 · ignorados (< medium): 1`. O histórico fica em `outputs/alerts/alerts.jsonl`, um alerta JSON por linha.

O cenário cobre os comportamentos do gerenciador:

- alerta repetido (mesmo paciente e tipo) dentro do cooldown é **suprimido**;
- piora de severidade (HIGH → CRITICAL) **sempre passa**, mesmo dentro do cooldown;
- evento abaixo de `MEDWATCH_ALERT_MIN_SEVERITY` é **ignorado**.

Fora do cenário, vale também: se nenhum canal conseguir entregar o alerta, o cooldown não é registrado e a próxima repetição é enviada normalmente. O webhook não recebe o `evidence` dos eventos (só tipo, score e horário); o registro completo fica no `alerts.jsonl`.

**Receber os alertas no celular/chat (opcional):** crie um webhook e configure no `.env`:

| Destino | Como criar | `MEDWATCH_ALERT_WEBHOOK_FORMAT` |
|---|---|---|
| Discord | Canal → Editar canal → Integrações → Webhooks → Novo webhook → Copiar URL | `discord` |
| Slack | App "Incoming Webhooks" → Add New Webhook to Workspace | `slack` |
| Teams | Canal → Workflows → "Post to a channel when a webhook request is received" | `teams` |

```dotenv
MEDWATCH_ALERT_WEBHOOK_URL=<url do webhook>
MEDWATCH_ALERT_WEBHOOK_FORMAT=discord
```

A URL do webhook dá acesso de escrita ao canal: mantenha-a só no `.env`.

## Áudio

Analisa a gravação de uma consulta e gera eventos `source=audio` para o `AlertManager`:

```bash
medwatch audio analyze consulta.wav --patient PAC-001 --start-time 2026-10-19T14:30:00-03:00
medwatch audio analyze consulta.wav --patient PAC-001 --alert        # envia os eventos aos canais
medwatch audio analyze consulta.wav --patient PAC-001 --diarize      # separa médico × paciente
medwatch audio analyze nova.wav --patient PAC-001 \
  --baseline outputs/reports/audio_PAC-001_<data>.json              # compara com a consulta anterior
```

| Etapa | Onde roda | O que faz |
|---|---|---|
| Pré-processamento | local | `.wav`, `.flac`, `.ogg` ou `.mp3` → PCM 16 kHz mono (taxa nativa do Azure Speech; 44,1 kHz só aumentaria o envio); valida tamanho e duração (3 s a 30 min) |
| Azure Speech to Text | Azure | Reconhecimento contínuo em pt-BR, com tempo por palavra e diarização opcional |
| Azure AI Language | Azure | Sentimento com opinion mining, frases-chave, resumo da consulta e Text Analytics for Health (sintomas, medicamentos e negação) |
| Termos críticos | local | `configs/critical_terms.yaml`: dispneia, fadiga, dor torácica, sintomas neurológicos e sangramento |
| Features acústicas | local | F0 (média e variação em semitons), jitter, shimmer, HNR, pausas, taxa de fala e de articulação |
| Indicadores | local | Fadiga, disartria e dificuldade respiratória por z-score contra o baseline (`configs/audio.yaml`) |

**Termos críticos.** Cada menção é classificada como afirmada, negada ou pergunta, e só as afirmadas geram evento:
- "Estou com falta de ar" é **afirmada**;
- "Não tenho falta de ar" e "Dor no peito, não" são **negadas**;
- "Sentiu falta de ar?" é **pergunta**, porque sem diarização é a fala do médico.

A negação vem do dicionário local e, quando disponível, da asserção do Text Analytics for Health.

**Indicadores são triagem, não diagnóstico.** Cada feature vira um z-score orientado (positivo = na direção do achado), e o indicador é a média ponderada desses z com piso 0: uma feature melhor que o baseline não compensa o desvio de outra. Com z ≥ 1,5 o indicador é `low`, com z ≥ 2 é `medium` e com z ≥ 3 é `high`. Um termo crítico afirmado da mesma família (ex.: "falta de ar" no indicador respiratório) sobe a severidade em um nível. O baseline padrão é uma referência populacional aproximada. Com `--baseline`, a comparação é com uma consulta anterior do próprio paciente, que é bem mais confiável.

**Resumo da consulta.** O padrão é o resumo **extrativo**: escolhe as frases mais representativas da própria consulta e mantém a ordem em que foram ditas. O resumo **abstrativo**, em que o modelo reescreve o texto, não está disponível em Brazil South. Para usá-lo, crie o recurso Language numa região suportada (ex.: East US, North Europe) e troque `summary_mode` em `configs/audio.yaml`. As frases-chave vêm do mesmo serviço.

**Saída.** O relatório fica em `outputs/reports/audio_<paciente>_<data>.json`, com transcrição, resumo, frases-chave, sentimento, entidades, termos, features, indicadores e eventos. Ele contém a transcrição completa: é dado de saúde e não deve ser anexado a issues, PRs ou mensagens. Os eventos e alertas carregam só metadados (categoria, termos do dicionário, contagens, z-scores); resumo e frases-chave ficam fora deles. O que vai para o Azure e o que fica local está em [docs/adr/0003-lgpd-audio-azure.md](docs/adr/0003-lgpd-audio-azure.md).

**Amostras e modo offline.**

```bash
medwatch audio synth-samples                    # gera data/samples/consulta_0*.wav com Azure TTS
medwatch audio analyze data/samples/consulta_02_dispneia.wav --patient PAC-001 --record
MEDWATCH_AZURE_MOCK=true medwatch audio analyze data/samples/consulta_02_dispneia.wav --patient PAC-001
```

- O `--record` grava as respostas do Azure em `data/samples/azure/<hash do áudio>.*.json`.
- Com `MEDWATCH_AZURE_MOCK=true`, essas respostas são reaproveitadas sem chamar o Azure (é o modo do CI).
- Grave assim **somente consultas fictícias**: esse diretório vai para o git.
- Para gravar com voz real, siga [docs/roteiro_gravacao_audio.md](docs/roteiro_gravacao_audio.md).
- Gravações em `.m4a` (iPhone) precisam ser convertidas antes: `ffmpeg -i consulta.m4a -ac 1 -ar 16000 consulta.wav`.

## Comandos de desenvolvimento

```bash
make lint     # ruff + mypy
make format   # aplica correções de estilo
make test     # testes sem Azure real e sem modelos pesados
```

## Estrutura

```
src/medwatch/   código (core, video, audio, anomaly, fusion, streaming, api, dashboard)
configs/        limiares, regras e pesos (YAML)
data/           datasets (raw/processed fora do git; samples versionados)
models/         pesos de modelos (fora do git)
outputs/        relatórios, alertas e logs (fora do git)
scripts/        provisionamento Azure e download de datasets
docs/           arquitetura, ADRs, relatório técnico
tests/          unit e integration
```
