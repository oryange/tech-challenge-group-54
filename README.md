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
| Sinais vitais (PR 3) | ⏳ |
| Áudio + Azure (PR 4) | ⏳ |
| Vídeo: YOLOv8 + OpenPose (PR 5) | ⏳ |
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
