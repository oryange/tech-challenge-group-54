# Roteiro do vídeo de demonstração (≤ 15 min)

Vídeo da entrega da Fase 4: YouTube ou Vimeo, **não listado**. O enunciado pede que ele mostre:
1. exemplo prático da análise de áudio e vídeo;
2. detecção e resposta a anomalias;
3. integração dos serviços Azure;
4. fluxo final do alerta à equipe médica.

O roteiro soma **13 min 30 s**, com 1 min 30 s de folga. Cada bloco indica o que mostrar na tela, o comando e a fala (resumida). A coluna "item" aponta qual dos 4 pontos do enunciado o bloco cobre.

## Antes de gravar

- [ ] `.env` com Speech, Language e **webhook real** (`MEDWATCH_ALERT_WEBHOOK_URL` + `MEDWATCH_ALERT_WEBHOOK_FORMAT`), e o canal aberto no celular ou numa janela ao lado
- [ ] `medwatch healthcheck --online` todo ✅ (o resultado aparece no bloco 3)
- [ ] Datasets baixados (`medwatch data download`, `--dataset videos`, `--dataset urfall`) e pesos em `models/`
- [ ] Rodar uma vez antes, para os modelos já estarem baixados e o terminal estar "quente":
  - `medwatch video analyze agachamento_unipodal.webm --exercise agachamento_unipodal`
  - `medwatch video analyze urfall_queda_01.mp4`
  - `medwatch vitals analyze --patient 10004235`
  - `medwatch prescriptions analyze --patient 10004235`
- [ ] Relatórios HTML de vídeo e PNGs de vitais e prescrições abertos em abas do navegador
- [ ] Portal Azure aberto em **rg-medwatch** (os dois recursos e a aba **Métricas**)
- [ ] Terminal com fonte grande (≥ 18 pt), tema claro e `clear` antes de cada bloco
- [ ] **Nenhuma chave na tela:** não abrir o `.env`; no portal, não clicar em "Mostrar as chaves"

## Blocos

| # | Tempo | Bloco | Item |
|---|---|---|---|
| 1 | 0:00–1:30 | Abertura e arquitetura | — |
| 2 | 1:30–4:00 | Vídeo: fisioterapia e queda | 1, 2 |
| 3 | 4:00–7:00 | Áudio com Azure | 1, 2, 3 |
| 4 | 7:00–9:00 | Sinais vitais e prescrições (dados reais) | 2 |
| 5 | 9:00–12:00 | Fusão em tempo real e alerta à equipe | 2, 4 |
| 6 | 12:00–13:30 | Resultados, limitações e encerramento | — |

### 1. Abertura e arquitetura (1 min 30 s)

- **Tela:** README no GitHub (título e tabela de status) e depois o diagrama de [architecture.md](architecture.md).
- **Fala:**
  - Apresentação do grupo e do problema: o hospital quer monitorar pacientes por vídeo, áudio e dados clínicos para identificar sinais precoces de risco.
  - **Fusão tardia:** cada módulo detecta anomalias na sua modalidade e emite eventos com o mesmo formato; o RiskEngine combina os eventos por paciente; o AlertManager entrega à equipe.
  - Datasets: MIMIC-IV demo (vitais e prescrições reais), UR Fall, vídeos do Wikimedia e consultas fictícias.

### 2. Vídeo: fisioterapia e queda (2 min 30 s)

```bash
medwatch video analyze agachamento_unipodal.webm --exercise agachamento_unipodal
```

- **Tela:**
  - saída do terminal com 3 repetições e "compensação com inclinação anterior do tronco";
  - abrir o **relatório HTML** (`outputs/reports/video/agachamento_unipodal_yolov8-pose.html`) e mostrar o quadro-chave com o esqueleto e o tronco a 69° contra o limite de 40°, e as recomendações;
  - mostrar 5 s do vídeo anotado.
- **Fala:** YOLOv8-pose estima 17 pontos do corpo; regras biomecânicas por exercício em `configs/video_rules.yaml`; o relatório sai automaticamente, com desvio, quadro-chave e recomendação.

```bash
medwatch video analyze agachamento_barra.webm --exercise agachamento --backend openpose
```

- **Tela:** a mesma análise com o **OpenPose** (rede original da CMU), com 2 flexões além do limite seguro.
- **Fala:** os dois backends ficam atrás da mesma interface e concordam em ~5 px.

```bash
medwatch video analyze urfall_queda_01.mp4
```

- **Tela:** o "Queda detectada" CRITICAL e o quadro-chave da pessoa no chão.
- **Fala:** a queda é a descida rápida do quadril seguida de postura horizontal. Na validação do UR Fall, foram 6/10 quedas detectadas com **zero alarmes falsos**.

### 3. Áudio com Azure (3 min)

```bash
medwatch healthcheck --online
```

- **Tela:** as linhas `azure.speech.online` e `azure.language.online` com ✅. Depois, o **portal Azure**: o grupo `rg-medwatch`, os dois recursos (Free F0, Brazil South) e a aba **Métricas** com as chamadas.
- **Fala:** Azure Speech to Text transcreve, e o Azure AI Language faz sentimento, frases-chave, resumo e Text Analytics for Health. Tudo em Brazil South (LGPD).

```bash
medwatch audio analyze data/samples/consulta_02_dispneia.wav --patient PAC-001 --start-time 2026-11-10T14:30:00-03:00
```

- **Tela:** tocar 10 s do áudio antes do comando. Depois, na saída, mostrar o resumo extrativo, os termos críticos **afirmados** (falta de ar, sem ar) × **negado** (dor no peito) e o evento `dyspnea_terms` MEDIUM.
- **Fala:**
  - Termos críticos respeitam a negação: "não tenho dor no peito" não gera evento.
  - As features acústicas (F0, jitter, shimmer, pausas) são calculadas localmente e viram indicadores de fadiga e disartria.
  - É **triagem, não diagnóstico**. Com o baseline de uma consulta anterior do próprio paciente, os indicadores ficam muito mais confiáveis.

### 4. Sinais vitais e prescrições, dados reais (2 min)

```bash
medwatch vitals analyze --patient 10004235
medwatch prescriptions analyze --patient 10004235
```

- **Tela:**
  - o PNG de vitais (`docs/img/vitals_paciente_10004235.png`): NEWS2 8 na admissão, SpO2 90%, e os episódios do Isolation Forest no fim, quando a FR sobe aos poucos;
  - o PNG de prescrições: potássio de 40 para 10 mEq (alto risco), heparina suspensa em 3 h, amiodarona de 600 para 150 e de volta a 600 mg.
- **Fala:**
  - Vitais com três técnicas complementares: NEWS2 (risco absoluto), z-score robusto (desvio do próprio paciente) e Isolation Forest (combinação rara na população, que pega a piora gradual).
  - Prescrições com regras explicáveis: salto de dose ≥ 3×, duplicidade e suspensão precoce, com peso maior para medicamentos de alto risco do ISMP.
  - Métricas com anomalias injetadas: F1 de 0,88 nos vitais e de 0,88–0,90 nas prescrições.

### 5. Fusão em tempo real e alerta à equipe (3 min)

**Terminal 1, 2 e 3** (ou `make up`):

```bash
medwatch api
medwatch dashboard
medwatch simulate --target api
```

- **Tela:** dividir a tela entre o **dashboard** (http://localhost:8501) e o **canal do webhook** (Teams ou Discord, no celular ou numa janela). Narrar enquanto o cenário roda (~40 s a 60×):

| Min do cenário | O que destacar |
|---|---|
| 3–10 | PAC-001: FC ↑ e cansaço, dois sinais fracos que nenhum alertaria sozinho, viram `fusion.risk_score`. Depois falta de ar + SpO2 88% vira **`fusion.respiratory_deterioration` CRITICAL** |
| 20–21 | PAC-002: o sensor e o vídeo veem a queda e geram **`fusion.fall_confirmed`** |
| 25 | PAC-004: a saída noturna do leito é um evento isolado e **não escala** |
| 40 | PAC-003: morfina triplicada e, 25 min depois, FR 8 irpm, o que gera **`fusion.adverse_drug_event`** |

- **No webhook:** o alerta CRITICAL chegando no celular, com paciente, motivos e ação recomendada ("Acionar a equipe de resposta rápida imediatamente").
- **No dashboard:** o ranking de pacientes por risco, a linha do tempo multimodal do PAC-002 e o clique em **ACK** em um alerta, que passa a aparecer como "reconhecido".
- **Fala:** a fusão tem dois mecanismos, as regras cross-modal e o risco combinado por noisy-OR. O cooldown evita repetir o mesmo alerta, mas uma piora sempre passa. A latência do evento ao alerta é de poucos milissegundos.

### 6. Resultados, limitações e encerramento (1 min 30 s)

- **Tela:** a tabela de resumo do [relatório técnico](relatorio_tecnico.md) (seção 1) e a seção de limitações.
- **Fala:**
  - Resultados por módulo, sempre lembrando que a precision é de anomalias injetadas, e não clínica.
  - Limitações:
    - fusão sem calibração com dados reais multimodais;
    - movimentação sintética;
    - consultas sintetizadas;
    - API sem autenticação, apenas local.
  - Próximos passos: calibração, diarização no áudio e autenticação.
  - Agradecimento e link do repositório.

## Depois de gravar

- [ ] Conferir se não aparece chave, `.env` nem dado pessoal de ninguém do grupo
- [ ] Duração ≤ 15 min
- [ ] Upload no YouTube como **não listado** (ou no Vimeo)
- [ ] Colar o link no README (seção do topo) e na entrega da FIAP
