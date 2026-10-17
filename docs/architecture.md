# Arquitetura — MedWatch

## Decisões principais

| Decisão | Escolha | Justificativa |
|---|---|---|
| Processamento de vídeo | Local/edge (OpenPose / YOLOv8) | Vídeo de paciente é dado sensível (LGPD); evita enviar imagem identificável à nuvem e reduz custo de banda. |
| Processamento de áudio | Azure Speech to Text + Azure AI Language, mais features acústicas locais | Requisito obrigatório; região `brazilsouth`. Pitch, pausas e jitter são extraídos localmente para não depender só do texto. |
| Estimativa de pose | Interface `PoseEstimator` com backends OpenPose (Docker/GPU) e YOLOv8-pose (padrão em dev) | OpenPose é difícil de instalar fora de Linux+CUDA. Ambos são normalizados para um esquema comum de keypoints. A demo e o relatório usam OpenPose. |
| Fusão | Late fusion orientada a eventos | Cada módulo emite `ClinicalEvent`s padronizados; o `RiskEngine` agrega por paciente numa janela deslizante. Módulos são desenvolvidos e testados isoladamente. |
| Tempo real | Simulador (replay acelerado de PhysioNet/MIMIC) → barramento assíncrono → FastAPI + WebSocket → Dashboard | Viável no escopo acadêmico; o barramento é uma interface e pode ganhar um adapter para Azure Event Hubs. |
| Nomenclatura Azure | "Cognitive Services" = Azure AI Services; Text Analytics faz parte do Azure AI Language | Registrado para evitar ambiguidade. |

## Diagrama

```
                         ┌──────────────────────── FONTES DE DADOS ─────────────────────────┐
                         │ Vídeos (fisio/quarto/cirurgia) │ Áudios consultas │ PhysioNet / MIMIC-IV demo │
                         └──────────┬─────────────────────┴────────┬─────────┴──────────┬────────────────┘
                                    │                              │                    │
                 ┌──────────────────▼──────────┐   ┌───────────────▼───────────┐  ┌─────▼─────────────────────┐
                 │  VIDEO PIPELINE (local)     │   │  AUDIO PIPELINE           │  │ STREAM SIMULATOR (replay) │
                 │  OpenCV frame sampler       │   │  ffmpeg → wav 16k mono    │  │ sinais vitais 1Hz         │
                 │  ├─ YOLOv8 (objetos/ROIs)   │   │  ├─ Acústica local        │  │ prescrições (eventos)     │
                 │  ├─ OpenPose | YOLOv8-pose  │   │  │  (librosa/parselmouth: │  └─────┬─────────────────────┘
                 │  ├─ Regras: ângulos,        │   │  │  F0, jitter, shimmer,  │        │
                 │  │  assimetria, queda, ROI  │   │  │  HNR, pausas)          │        │
                 │  └─ Série temporal de pose ─┼─┐ │  ├─ Azure Speech to Text ─┼──┐     │
                 │  Relatório (HTML/JSON + PNG)│ │ │  └─ Azure AI Language  ◄──┼──┘     │
                 └──────────────┬──────────────┘ │ │     (Sentiment, Health,   │        │
                                │                │ │      termos críticos)     │        │
                                │                │ └────────────┬──────────────┘        │
                                │      ┌─────────▼──────────────┼───────────────────────▼──────────┐
                                │      │            ANOMALY DETECTION                              │
                                │      │  vitals: limiares clínicos (NEWS2) + z-score robusto      │
                                │      │          + Isolation Forest (janelas)                     │
                                │      │  prescrições: regras (dose Δ, alto risco, duplicidade)    │
                                │      │          + Isolation Forest (paciente-dia)                │
                                │      │  movimentação: imobilidade, saída do leito, queda         │
                                │      └─────────┬─────────────────────────────────────────────────┘
                                ▼                ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │  EVENT BUS (asyncio; adapter p/ Azure Event Hubs opcional) │
                 │  ClinicalEvent{patient_id, source, type, score, severity,  │
                 │                evidence, artifacts, timestamp}             │
                 └──────────────────────────┬────────────────────────────────┘
                                            ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │  FUSION / RISK ENGINE                                      │
                 │  janela deslizante por paciente · noisy-OR ponderado       │
                 │  regras de escalonamento cross-modal (ex.: "falta de ar"   │
                 │  no áudio + queda de SpO2 em 10 min → CRITICAL)            │
                 │  dedup + cooldown                                          │
                 └──────────────────────────┬────────────────────────────────┘
                                            ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │  ALERT MANAGER → console/log · alerts.jsonl · webhook      │
                 │                  (Teams/Slack) · WebSocket                 │
                 └──────────────────────────┬────────────────────────────────┘
                                            ▼
                     FastAPI (/ingest, /alerts, /ws/alerts)  ──►  Dashboard Streamlit (equipe médica)
```

## Contratos de dados (implementados no PR 2)

- **ClinicalEvent**: `event_id` (uuid), `patient_id` (pseudonimizado), `source` (`video` | `audio` | `vitals` | `prescription` | `movement`), `type`, `timestamp` (UTC), `score` ∈ [0, 1], `severity` (`info` < `low` < `medium` < `high` < `critical`), `evidence` (dict serializável), `artifacts` (caminhos de relatórios/frames).
- **Alert**: `alert_id`, `patient_id`, `severity`, `title`, `reasons` (eventos contribuintes), `recommended_action`, `created_at`, `acked`.

Score agregado de risco por paciente na janela: `R = 1 − Π(1 − wₛ · scoreᵢ)`, com pesos `wₛ` por fonte em `configs/fusion.yaml`.

## Datasets

| Uso | Dataset | Observação |
|---|---|---|
| Sinais vitais (BP, HR, SpO2, RR) e prescrições | MIMIC-IV Clinical Database Demo (PhysioNet, acesso aberto, 100 pacientes) | Tabelas `chartevents`, `prescriptions`, `emar`. |
| Vitais de alta frequência para streaming | BIDMC PPG and Respiration (PhysioNet) | HR, RR, SpO2 a 1 Hz. |
| Sons respiratórios | AudioSet (Breathing, Cough, Wheeze, Gasp) | Complemento da análise acústica. |
| Consultas em pt-BR | Gravações roteirizadas pelo grupo (normal × fatigado/dispneico) | Gera ground truth. |
| Vídeo | Gravações do grupo (exercícios corretos/incorretos, simulação de queda) e vídeos com licença aberta | Pesos COCO do YOLOv8 não detectam instrumental cirúrgico; cirurgia é stretch goal com fine-tune. |

Avaliação: anomalias sintéticas rotuladas são injetadas em vitais e prescrições para medir precision/recall/F1.

## Privacidade (LGPD)

- Vídeo nunca sai do ambiente local.
- Áudio é enviado ao Azure em `brazilsouth`; transcrições completas não vão para logs.
- Identificadores de paciente são pseudonimizados em eventos, alertas e logs.
