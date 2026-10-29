# Plano de Desenvolvimento e Checklists de PR

## Fluxo Git

- `main` protegida: exige 1 review e CI verde.
- Branches `feat/pr-XX-<slug>`, squash merge, Conventional Commits (`feat:`, `fix:`, `docs:`...).

## Sequência de PRs

| PR | Escopo | Depende de | Paralelo com |
|---|---|---|---|
| 1 | Setup, infraestrutura, config, logging, CLI, Docker, CI, provisionamento Azure | — | — |
| 2 | Core: esquema de eventos, EventBus, AlertManager e canais | 1 | — |
| 3 | Anomalias em sinais vitais (MIMIC-IV demo/BIDMC) | 2 | 4, 5 |
| 4 | Áudio: Azure STT, Azure Language, features acústicas | 2 | 3, 5 |
| 5 | Vídeo: YOLOv8, pose (OpenPose/YOLOv8-pose), regras, relatório | 2 | 3, 4 |
| 6 | Anomalias em prescrições e padrões de movimentação | 3, 5 | — |
| 7 | Fusão (RiskEngine), simulador em tempo real, API e Dashboard | 3–6 | — |
| 8 | Relatório técnico, roteiro da demo, hardening final | 7 | — |

Comece o PR 4 cedo: provisionar o Azure (assinatura, F0, termos de IA Responsável) costuma travar.

---

## Definition of Done global (todos os PRs)

- [ ] Branch `feat/pr-XX-*` atualizada com `main` e sem conflitos
- [ ] `ruff check` e `ruff format --check` limpos; `mypy src` sem erros novos
- [ ] `pytest -m "not azure and not slow"` verde no CI; cobertura do código novo ≥ 70%
- [ ] Nenhum segredo commitado (`.env`, chaves, endpoints com key); `.env.example` atualizado
- [ ] Nenhum dado bruto, peso de modelo ou arquivo > 5 MB versionado (exceto `data/samples/`)
- [ ] Dependência nova declarada no `requirements.txt` com piso e teto no próximo major, e comentário quando o motivo não for óbvio
- [ ] Docstrings nas funções públicas; parâmetros mágicos em `configs/*.yaml`
- [ ] Logs estruturados sem PII (`patient_id` pseudonimizado, nunca nome)
- [ ] README atualizado (seção do módulo e comando de execução)
- [ ] Descrição do PR com: o que mudou, como testar, evidências (log, screenshot, JSON)
- [ ] 1 aprovação de outro membro do grupo

---

## PR 1 — Setup e infraestrutura

**Código**
- [ ] Layout `src/medwatch/` instalável (`pip install -e .`) com entrypoint `medwatch`
- [ ] `config.py` com `pydantic-settings`: prefixos `MEDWATCH_` e `AZURE_`, chaves como `SecretStr`, `ensure_dirs()`
- [ ] `log.py`: saída texto/JSON configurável e arquivo rotativo em `outputs/logs/`
- [ ] `cli.py` com `medwatch healthcheck` (offline, `--online` para Azure, exit code ≠ 0 se algo obrigatório falhar)
- [ ] Esqueleto dos pacotes (`core`, `video`, `audio`, `anomaly`, `fusion`, `api`, `dashboard`)

**Dependências / Infra**
- [ ] `requirements.txt` único (piso + teto no próximo major, comentários) terminando em `-e .`; `pyproject.toml` só com metadados e config de ferramentas
- [ ] `docker/Dockerfile` (python 3.11 slim-bookworm, torch CPU, ffmpeg, libs do Speech SDK, usuário não-root) builda sem erro
- [ ] `docker-compose.yml` com volumes `data/`, `models/`, `outputs/` e `.env`
- [ ] `scripts/provision_azure.sh` idempotente (RG, Speech F0, Language F0, imprime variáveis)
- [ ] `.gitignore`, `.env.example`, `.pre-commit-config.yaml`, `Makefile`
- [ ] `.github/workflows/ci.yml` e `.github/pull_request_template.md`

**Testes**
- [ ] `test_config.py`: override por env, Azure não configurado sem chaves, `ensure_dirs` em `tmp_path`
- [ ] `test_cli.py`: `healthcheck` sai com 0 sem Azure; `--strict` sai com 1
- [ ] `docker compose run --rm medwatch healthcheck` executa

**README**
- [ ] Visão geral, pré-requisitos, setup local e Docker, provisionamento Azure, tabela de variáveis

**Validação**
- [ ] `medwatch healthcheck` imprime a tabela ✅/❌ e grava `outputs/logs/medwatch.log`
- [ ] `medwatch healthcheck --online` com `.env` real retorna ✅ em `azure.language.online` (evidência no PR, chaves mascaradas)

## PR 2 — Core: eventos, barramento e alertas

**Código**
- [x] `core/events.py`: `Source`, `Severity` (ordenável), `ClinicalEvent`, `Alert` (pydantic, uuid, timestamps UTC timezone-aware)
- [x] `score` validado em [0, 1]; `evidence` serializável; `artifacts: list[Path]`
- [x] `core/bus.py`: protocolo `EventBus` (`publish`, `subscribe`) e `InMemoryEventBus` (asyncio.Queue, fan-out)
- [x] `core/alerts/`: `AlertManager` com dedup (paciente+tipo), cooldown configurável e canais `ConsoleChannel`, `JsonlChannel` (`outputs/alerts/alerts.jsonl`), `WebhookChannel` (Teams/Slack, timeout e retry)
- [x] Falha num canal não derruba os outros

**Dependências**
- [x] `httpx` no core; nenhuma dependência pesada

**Testes**
- [x] Round-trip de serialização de `ClinicalEvent` e `Alert`
- [x] Bus: dois subscribers recebem o mesmo evento
- [x] AlertManager: dedup no cooldown, reenvio depois, canal com exceção não bloqueia os demais
- [x] Webhook com `httpx.MockTransport` (sem rede)

**Docs**
- [x] `docs/architecture.md` atualizado; `docs/adr/0001-late-fusion-event-driven.md`

**Validação**
- [ ] `medwatch alerts demo` emite alerta fake no console, em `alerts.jsonl` (JSON válido por linha) e no webhook de teste (screenshot)

## PR 3 — Anomalias em sinais vitais

**Código**
- [ ] `medwatch data download` baixa o MIMIC-IV demo, verifica SHA-256 e é idempotente (BIDMC fica para o simulador do PR 7)
- [ ] `anomaly/detectors.py`: `ThresholdRule`, `RobustZScore` (mediana/MAD por paciente, janela rolante), `IsolationForestDetector` (features: valor do momento e variação de 1 h; média, desvio e slope em janela ficaram como evolução, ver `docs/relatorio_vitais.md`)
- [ ] `anomaly/vitals.py`: normaliza itemids MIMIC (HR, SBP, DBP, SpO2, RR, Temp), calcula NEWS2, emite `ClinicalEvent(source=vitals)`
- [ ] Limiares em `configs/thresholds.yaml` com referência clínica
- [ ] Tratamento de gaps e artefatos de sensor (SpO2 = 0 é artefato)

**Dependências**
- [ ] Seção "Anomalias" do `requirements.txt` revisada (`scikit-learn`, `wfdb`, `pandas`)

**Testes**
- [ ] Spike sintético detectado; série estável sem evento
- [ ] NEWS2 parametrizado com casos conhecidos
- [ ] Artefato filtrado
- [ ] `@slow` em paciente real do demo

**Docs**
- [ ] Seção "Sinais vitais" no README: dataset, download, `medwatch vitals analyze --patient <id>`

**Validação**
- [ ] Precision/recall/F1 sobre anomalias injetadas
- [ ] Gráfico com anomalias marcadas em `outputs/reports/vitals_<id>.png`
- [ ] Severidade dos eventos coerente com NEWS2

## PR 4 — Áudio: Azure Speech + Azure Language + acústica

**Código**
- [ ] `preprocessing.py`: qualquer formato → wav PCM 16 kHz mono (ffmpeg), valida duração
- [ ] `azure_stt.py`: reconhecimento contínuo, `pt-BR`, timestamps por palavra, diarização opcional (`ConversationTranscriber`)
- [ ] `azure_language.py`: Sentiment com opinion mining, Text Analytics for Health (sintomas, medicamentos, negação), batching nos limites da API; confirmar suporte a pt-BR no Health, com fallback no dicionário
- [ ] `critical_terms.py`: `configs/critical_terms.yaml` com normalização e respeito à negação
- [ ] `acoustic.py`: F0 (média/variância), jitter, shimmer, HNR (parselmouth), taxa de fala/articulação, pausas
- [ ] Indicadores de fadiga, disartria e dificuldade respiratória via z-score contra baseline, documentados como triagem
- [ ] Retry com backoff (429/5xx), timeout, modo mock (`MEDWATCH_AZURE_MOCK=true`)
- [ ] Nenhum áudio ou transcrição com PII em log

**Dependências / Infra**
- [ ] Seção "Áudio" do `requirements.txt` revisada; Dockerfile com `libasound2`, `libssl3`, `ca-certificates`

**Testes**
- [ ] Mocks dos clientes Azure (fixtures em `tests/fixtures/azure/`)
- [ ] Features acústicas em tom sintético (F0 ±5%)
- [ ] "não tenho falta de ar" não dispara termo crítico
- [ ] `@azure`: transcrição real de `data/samples/consulta_01.wav`

**Docs**
- [ ] Seção "Áudio": `medwatch audio analyze <arquivo>` e explicação das features
- [ ] ADR de LGPD (o que vai para o Azure × local)

**Validação**
- [ ] `outputs/reports/audio_<id>.json` com transcrição, sentimento, entidades, termos críticos e features
- [ ] Consulta fatigada/dispneica gera evento ≥ medium; a normal não
- [ ] Evidência: JSON e print das métricas do recurso no portal Azure

## PR 5 — Vídeo: YOLOv8 + OpenPose

**Código**
- [ ] `video/pose/keypoints.py`: esquema comum (BODY_25 ↔ COCO-17) com confiança por ponto
- [ ] `PoseEstimator` (ABC), `YoloPoseEstimator` (padrão), `OpenPoseEstimator` (rede da CMU em PyTorch) e leitura dos JSON do binário oficial
- [ ] `objects.py`: YOLOv8 com tracking (ByteTrack) e ROIs críticas em `video_rules.yaml`
- [ ] `rules.py`: ângulos vs. faixa esperada, assimetria, compensação de tronco, queda (velocidade do quadril + razão de aspecto), invasão de ROI
- [ ] Amostragem de frames configurável e suavização temporal dos keypoints
- [ ] `report.py`: relatório HTML e JSON (timeline, frames anotados, % fora do padrão, recomendações)
- [ ] Exporta série temporal de pose (`parquet`) para o PR 6

**Dependências / Infra**
- [ ] Seção "Vídeo" do `requirements.txt` revisada; pesos baixados em `models/` sob demanda
- [ ] Decisão sobre o OpenPose registrada em ADR (OpenCV 5 sem Caffe; rede portada para PyTorch) e avisos de licença em `THIRD_PARTY_NOTICES.md`

**Testes**
- [ ] Ângulos com keypoints sintéticos (90°, 180°)
- [ ] Mapeamento BODY_25 → COCO
- [ ] Queda vs. sentar devagar com sequência sintética
- [ ] `@slow`: clipe de 5 s em `data/samples/`

**Docs**
- [ ] Seção "Vídeo": `medwatch video analyze <video> --exercise agachamento --backend yolo|openpose`; ADR do backend dual

**Validação**
- [ ] Vídeo anotado e relatório HTML abrindo no browser
- [ ] Execução incorreta gera desvios; correta gera pouquíssimos
- [ ] Ao menos uma execução com OpenPose registrada
- [ ] FPS medido (CPU vs. GPU)

## PR 6 — Prescrições e movimentação

**Código**
- [x] `anomaly/prescriptions.py`: timeline por paciente; regras (Δ dose ≥ 3× em 24 h, droga de alto risco, duplicidade, suspensão precoce) + Isolation Forest paciente-dia
- [x] Lista de alto risco em `configs/prescriptions.yaml` (base ISMP)
- [x] Anomalias rotuladas injetadas nas prescrições reais (`anomaly/prescription_evaluation.py`, no lugar de um gerador separado)
- [x] `anomaly/movement.py`: atividade/min, imobilidade prolongada, saída do leito noturna, quedas (+ queda de atividade e adaptador da série de pose do vídeo)
- [x] `evidence` explicável (regra, valores antes/depois)

**Testes**
- [x] Cada regra com caso positivo e negativo
- [x] Conversão mg/g sem falso positivo
- [x] Série parada > N min dispara imobilidade

**Docs**
- [x] Seções "Prescrições" e "Movimentação" (README e `docs/relatorio_prescricoes_movimentacao.md`)

**Validação**
- [x] P/R/F1 nas anomalias injetadas
- [x] Exemplos de alerta de prescrição e imobilidade em `alerts.jsonl`

## PR 7 — Fusão, tempo real, API e Dashboard

**Código**
- [x] `fusion/risk_engine.py`: janela deslizante, noisy-OR ponderado, regras cross-modal em `configs/fusion.yaml`
- [x] `Alert.reasons` com eventos contribuintes
- [x] `streaming/simulator.py`: replay com `--speed` e cenários roteirizados
- [x] FastAPI: `POST /ingest/{audio,video}`, `GET /patients/{id}/risk`, `GET /alerts`, `POST /alerts/{id}/ack`, `WS /ws/alerts`, `GET /health` (mais `POST /ingest/events` e `GET /patients`)
- [x] Streamlit: pacientes por risco, timeline multimodal, feed em tempo real, ACK
- [x] `docker-compose` com serviços `api` e `dashboard` (e `simulator` no profile `demo`)

**Testes**
- [x] Evento isolado não escala; combinação cross-modal → CRITICAL
- [x] API com `TestClient` (inclui WebSocket)
- [x] Integração simulador → bus → risk → alert em < 2 s

**Docs**
- [x] `make demo`; seção "Fluxo de alerta" com diagrama de sequência

**Validação**
- [x] Cenário de demo produz a sequência de alertas documentada
- [x] Latência evento→alerta medida (mediana 3,5 ms, máx. 5,7 ms)
- [x] Screenshots do dashboard
- [ ] `make up` validado com Docker

## PR 8 — Relatório técnico, demo e hardening

- [ ] `docs/relatorio_tecnico.md` (e PDF): fluxo multimodal, modelos por tipo de dado, datasets, métricas, exemplos reais de anomalias, limitações, LGPD/ética, trabalhos futuros
- [ ] `docs/roteiro_video.md` (≤ 15 min): arquitetura (2), vídeo (3), áudio/Azure (3), vitais/prescrição (2), fusão e alerta (3), conclusão (2)
- [ ] README final: badge do CI, quickstart, link do vídeo (YouTube não listado)
- [ ] Clone limpo + `make install && make demo` testado por quem não escreveu o código
- [ ] Custo Azure verificado e teardown documentado
- [ ] Tag `v1.0.0` e release no GitHub
