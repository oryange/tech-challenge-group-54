# Roteiro de gravação: MedWatch (Tech Challenge Fase 4, Grupo 54)

Duração-alvo: **11 min** (mínimo 10, máximo 12).

| Parte | Quem | Tempo | Conteúdo |
|---|---|---|---|
| 1 | Ory | 0:00 – 5:30 | Abertura, problema, arquitetura, vídeo (OpenPose/YOLOv8), áudio (Azure) |
| 2 | Laris | 5:30 – 11:00 | Sinais vitais, prescrições, movimentação, fusão, demo em tempo real, conclusão |

Como ler cada bloco:
- **TELA:** o que precisa estar visível. O formato `arquivo:linha` significa abrir o arquivo no VS Code e **selecionar** as linhas indicadas.
- **AÇÃO:** o que fazer enquanto fala.
- **FALA:** o texto para ler. Ritmo de ~130 palavras/min.

Todos os caminhos são relativos à raiz do repositório `tech-challenge-group-54`.

---

## Preparação (antes de gravar, não aparece no vídeo)

1. Preencher `AZURE_SPEECH_KEY` e `AZURE_LANGUAGE_KEY` no `.env` e rodar `medwatch healthcheck --online` até ficar tudo ✅. Sem isso, o bloco A5 não funciona ao vivo.
2. Limpar o canal do Discord que recebe o webhook, para aparecerem só os alertas da gravação.
3. Gerar antes os artefatos de vídeo, que demoram para processar:
   ```bash
   medwatch video analyze agachamento_unipodal --exercise agachamento_unipodal --backend yolo
   medwatch video analyze urfall_queda_01 --backend openpose
   ```
   Confira que existem `outputs/reports/video/agachamento_unipodal_yolov8-pose.html` e `outputs/reports/video/urfall_queda_01_openpose_anotado.mp4`.
4. Gerar antes o relatório de vitais: `medwatch vitals analyze --patient 10004235`.
5. Telas:
   - VS Code com fonte 16–18 e a barra lateral fechada;
   - terminal com fonte 18 e o venv ativo;
   - navegador com três abas: `http://localhost:8501`, o Discord e o portal do Azure (página de **Métricas** do recurso Speech).
6. Laris: antes da Parte 2, deixar abertos 3 terminais:
   - **T1:** `medwatch api`, já rodando;
   - **T2:** `medwatch dashboard`, já rodando;
   - **T3:** livre, para o simulador.
7. Fechar notificações (Slack, e-mail) e esconder o `.env`. **Nunca** abrir o `.env` na tela.

---

# PARTE 1: Ory (0:00 – 5:30)

### A1. Abertura (0:00 – 0:30)

**TELA:** `README.md`, topo (linhas 1–10), em modo preview no VS Code (`Cmd+Shift+V`).

**FALA:**
> "Olá! Somos o Grupo 54 da Pós-Tech de IA para Devs, e este é o MedWatch, nosso Tech Challenge da Fase 4. O MedWatch é um sistema de monitoramento multimodal de pacientes. Ele analisa vídeo, áudio, sinais vitais, prescrições e movimentação, identifica sinais precoces de risco e alerta a equipe médica em tempo real. Eu vou apresentar a arquitetura e os módulos de vídeo e áudio, e depois passo para a minha colega, que mostra os dados clínicos, a fusão e a demonstração ao vivo."

---

### A2. Problema e objetivos (0:30 – 1:00)

**TELA:** `docs/relatorio_tecnico.md:36-45` (seção 2, "Problema e objetivos") em preview.

**FALA:**
> "O enunciado pede quatro coisas: analisar vídeos de cirurgia ou fisioterapia em busca de padrões anômalos; processar a voz das consultas para detectar sintomas como fadiga e disartria; detectar anomalias em sinais vitais e prescrições; e usar serviços gerenciados de nuvem, no nosso caso o Azure. O objetivo final é juntar tudo isso num alerta único e explicável para a equipe."

---

### A3. Arquitetura: fusão tardia orientada a eventos (1:00 – 2:00)

**TELA 1 (1:00 – 1:35):** `docs/relatorio_tecnico.md:49-66`, o diagrama ASCII da seção 3. Selecione o bloco inteiro.

**AÇÃO:** passe o cursor de cima para baixo enquanto fala: módulos → `ClinicalEvent` → `EventBus` → `RiskEngine` → `AlertManager` → canais.

**FALA:**
> "Esta é a arquitetura. No topo estão as cinco fontes de dados. Cada uma tem seu próprio módulo de detecção e todas emitem o mesmo contrato, o ClinicalEvent. Os eventos passam por um barramento assíncrono. O RiskEngine faz a fusão por paciente, numa janela de tempo, e o AlertManager aplica severidade mínima, deduplicação e cooldown antes de entregar no console, num histórico, no webhook e no dashboard."

**TELA 2 (1:35 – 2:00):** `src/medwatch/core/events.py:130-143`, a classe `ClinicalEvent`.

**AÇÃO:** destaque as linhas 136 (`patient_id`), 140 (`score`), 141 (`severity`) e 142 (`evidence`).

**FALA:**
> "O contrato é este: paciente pseudonimizado, fonte, tipo, horário, um score de zero a um, a severidade e a evidência que explica o porquê. Como a fusão trabalha sobre eventos e não sobre dados brutos, fica simples combinar um vídeo a dez quadros por segundo com prescrições diárias."

---

### A4. Vídeo: YOLOv8 e OpenPose (2:00 – 3:45)

**TELA 1 (2:00 – 2:25):** `docs/relatorio_tecnico.md:104-111`, a tabela da seção 5.1.

**FALA:**
> "No vídeo usamos dois backends de pose: o YOLOv8-pose, que é o padrão por ser rápido em CPU, e o OpenPose da CMU, que é o requisito do enunciado. Portamos a rede original do OpenPose para PyTorch, e os keypoints dos dois concordam em cerca de cinco pixels. Sobre os keypoints rodam regras biomecânicas, e o YOLOv8 com ByteTrack detecta objetos e áreas críticas."

**TELA 2 (2:25 – 2:45):** `configs/video_rules.yaml:58-63`, a regra `inclinacao_tronco_excessiva` do agachamento unipodal.

**AÇÃO:** destaque a linha 60 (`max: 40`).

**FALA:**
> "As regras ficam em YAML, com limite, mensagem e recomendação. Aqui, no agachamento unipodal, uma inclinação do tronco acima de quarenta graus é marcada como compensação."

**TELA 3 (2:45 – 3:15):** navegador com `outputs/reports/video/agachamento_unipodal_yolov8-pose.html` aberto (arraste o arquivo para o navegador).

**AÇÃO:** role até os achados e mostre o quadro-chave e o gráfico de métricas.

**FALA:**
> "Este é o relatório gerado automaticamente para um vídeo real de fisioterapia. O sistema contou três repetições e, nas três, encontrou compensação com o tronco inclinado até sessenta e nove graus, contra um limite de quarenta. Cada achado vem com o quadro-chave e a recomendação para o fisioterapeuta."

**TELA 4 (3:15 – 3:45):** `outputs/reports/video/urfall_queda_01_openpose_anotado.mp4` tocando (QuickTime). Depois, `docs/relatorio_tecnico.md:189-192`, a tabela de quedas.

**FALA:**
> "Para quedas, usamos o dataset UR Fall. A regra é: o quadril desce pelo menos vinte e cinco por cento da altura em até um segundo, seguido de postura horizontal. Calibramos nas sequências de um a dez e validamos nas de onze a vinte: o YOLOv8 detectou seis de dez quedas com zero alarmes falsos. Priorizamos não gerar alarme falso, porque alarme demais faz a equipe ignorar o sistema."

---

### A5. Áudio: Azure Speech e Azure Language (3:45 – 5:15)

**TELA 1 (3:45 – 4:10):** `docs/relatorio_tecnico.md:117-124`, a tabela da seção 5.2.

**FALA:**
> "No áudio da consulta, o Azure Speech to Text faz a transcrição em português, na região Brazil South. O Azure AI Language analisa o texto: sentimento, frases-chave, resumo e o Text Analytics for Health. Localmente, procuramos termos críticos e medimos a voz: frequência, jitter, shimmer, pausas e taxa de fala, para indicar fadiga e disartria."

**TELA 2 (4:10 – 4:55):** terminal. Rode, ao vivo e **com o Azure real** (sem a variável de mock):

```bash
medwatch audio analyze data/samples/consulta_02_dispneia.wav --patient PAC-001
```

**AÇÃO:** enquanto processa (~20–30 s), fale o primeiro parágrafo. Quando sair o resultado, aponte o resumo, os termos afirmados, o "dor no peito" negado e os eventos.

**FALA:**
> "Vou rodar agora, ao vivo, uma consulta em que a paciente relata falta de ar. O áudio vai para o Azure, que devolve a transcrição e a análise do texto."
>
> "Aqui está: o resumo extrativo do Azure, os termos críticos afirmados, como 'falta de ar' e 'sem ar', e repare que 'dor no peito' aparece como **negado**, porque a paciente disse que não tinha. Termo negado não gera evento. No fim, os eventos: dispneia como MEDIUM e o indicador de fadiga."

**TELA 3 (4:55 – 5:15):** navegador, aba do portal Azure, **Métricas** do recurso Speech (ou a imagem `docs/img/audio_azure_metricas_speech.png`).

**FALA:**
> "No portal, dá para ver as chamadas chegando ao recurso Speech, no plano gratuito. Sobre privacidade: o vídeo nunca sai da máquina, o áudio fica no Brasil e todas as chamadas ao Language desligam o log do serviço. Os alertas levam só metadados, nunca a transcrição."

---

### A6. Passagem (5:15 – 5:30)

**TELA:** `docs/relatorio_tecnico.md:47`, o título "3. Fluxo multimodal", com o diagrama visível.

**FALA:**
> "Esses são os sinais de vídeo e de áudio. Agora a Laris mostra os dados clínicos e, principalmente, como tudo isso se junta num alerta."

---

# PARTE 2: Laris (5:30 – 11:00)

### B1. Sinais vitais com dados reais (5:30 – 6:45)

**TELA 1 (5:30 – 5:55):** `configs/thresholds.yaml:23-28`, as faixas do NEWS2.

**FALA:**
> "Obrigada! Nos sinais vitais usamos o MIMIC-IV, com dados reais de UTI. São três técnicas que se complementam. O NEWS2, que é o escore clínico do Royal College of Physicians, aqui com as faixas de cada parâmetro, responde se o valor é perigoso. O z-score robusto compara o paciente com ele mesmo. E o Isolation Forest procura combinações raras na população."

**TELA 2 (5:55 – 6:25):** imagem `outputs/reports/vitals_10004235_34100191.png` aberta (ou `docs/img/vitals_paciente_10004235.png`).

**AÇÃO:** aponte a admissão (início do gráfico) e depois a subida da FR no final.

**FALA:**
> "Este é um paciente real, o 10004235. Na admissão, o NEWS2 chega a oito, que é crítico, com SpO2 de noventa por cento. No final da internação, a frequência respiratória sobe devagar de vinte para mais de quarenta. O z-score não pega essa piora, porque o baseline vai subindo junto, e quem sinaliza é o Isolation Forest. Por isso usamos as três técnicas."

**TELA 3 (6:25 – 6:45):** `docs/relatorio_tecnico.md:233-238`, a tabela de métricas de vitais.

**AÇÃO:** destaque a linha `Combinado`.

**FALA:**
> "Como o dado real não tem rótulo de anomalia, injetamos anomalias conhecidas em cento e dezesseis internações reais. Combinadas, as três técnicas chegam a F1 de 0,88."

---

### B2. Prescrições e movimentação (6:45 – 7:45)

**TELA 1 (6:45 – 7:10):** `configs/prescriptions.yaml:64-72`, a regra `dose_change`.

**AÇÃO:** destaque a linha 70 (`increase_ratio: 3.0`) e o comentário das linhas 65–67.

**FALA:**
> "Nas prescrições, também do MIMIC, detectamos salto de dose, duplicidade terapêutica e suspensão precoce. Os limiares saíram dos dados. Com o salto de duas vezes seriam quase mil eventos, porque dobrar a dose é rotina. Com três vezes caem para cento e sessenta. Medicamentos de alto risco da lista do ISMP sobem a severidade."

**TELA 2 (7:10 – 7:45):** `configs/movement.yaml:15-30`, as regras de movimentação.

**AÇÃO:** destaque as linhas 17–19 (imobilidade, 120 min) e 23–26 (saída noturna).

**FALA:**
> "Na movimentação no leito, os dados vêm de um sensor ou da própria pose do vídeo. Detectamos imobilidade prolongada a partir de duas horas, que é a referência de reposicionamento para prevenir lesão por pressão, saída do leito à noite e suspeita de queda. O sensor sozinho confunde uma ida brusca à poltrona com uma queda, e é aí que entra a fusão."

---

### B3. Fusão multimodal (7:45 – 8:45)

**TELA 1 (7:45 – 8:20):** `configs/fusion.yaml:31-50`, as regras `respiratory_deterioration` e `fall_confirmed`.

**AÇÃO:** destaque as linhas 32–40 e depois 42–50.

**FALA:**
> "A fusão tem regras cross-modal. A piora respiratória junta a queixa de falta de ar na consulta com a queda de saturação nos vitais, numa janela de trinta minutos, e vira crítica. A queda confirmada exige o vídeo **e** o sensor em até cinco minutos, o que resolve o problema da poltrona. Também temos efeito adverso de medicamento, em que a piora precisa vir **depois** da mudança da prescrição."

**TELA 2 (8:20 – 8:45):** `src/medwatch/fusion/risk_engine.py:58-63`, a função `noisy_or`. Depois, `configs/fusion.yaml:11-16`, os pesos.

**FALA:**
> "Além das regras, calculamos um risco combinado por noisy-OR ponderado. Cada modalidade tem um peso: vitais valem um, o vídeo 0,9, e o áudio e as prescrições menos, porque são indícios. Assim, dois sinais fracos de fontes diferentes podem somar um alerta que nenhum deles daria sozinho."

---

### B4. Demonstração em tempo real (8:45 – 10:15)

**TELA 1 (8:45 – 8:55):** terminal T3.

**AÇÃO:** rode:
```bash
medwatch simulate --target api
```
Mude imediatamente para o navegador, na aba `http://localhost:8501`.

**FALA:**
> "Agora a demonstração ao vivo. O simulador envia para a API quarenta minutos de internação de quatro pacientes, acelerados: cada minuto vira um segundo."

**TELA 2 (8:55 – 9:50):** dashboard em `http://localhost:8501`.

**AÇÃO:** comente conforme os alertas aparecem. A ordem é fixa:
- ~3 s: PAC-001, risco combinado;
- ~10 s: PAC-001, piora respiratória CRITICAL;
- ~15 s: PAC-003, dose;
- ~21 s: PAC-002, queda confirmada;
- ~30 s: PAC-002, instabilidade pós-queda;
- ~40 s: PAC-003, efeito adverso.

**FALA (acompanhe os alertas):**
> "A PAC-001 começa com a frequência cardíaca um pouco alta e se queixa de cansaço. Sozinhos, nenhum dos dois alertaria, mas juntos geram risco médio. Ela relata falta de ar e a saturação cai para oitenta e oito: piora respiratória, crítico."
>
> "A PAC-003 tem a morfina triplicada. A PAC-002 cai: o vídeo e o sensor confirmam a queda, e logo depois a taquicardia gera o alerta de instabilidade pós-queda."
>
> "No final, a frequência respiratória da PAC-003 cai para oito, vinte e cinco minutos depois do aumento do opioide: possível efeito adverso, HIGH. A PAC-004 teve só uma saída noturna isolada, que não escala. No painel, os pacientes ficam ordenados por risco."

**AÇÃO (9:40):** clique em **ACK** num alerta crítico e diga:
> "A equipe reconhece o alerta por aqui."

**TELA 3 (9:50 – 10:15):** navegador, aba do Discord.

**FALA:**
> "Os mesmos alertas chegam no canal da equipe pelo webhook, aqui no Discord, mas o mesmo funciona para Teams e Slack. A mensagem leva só o resumo: tipo, score e horário, sem dados clínicos detalhados. Do evento ao alerta, a latência fica em milissegundos."

---

### B5. Limitações e conclusão (10:15 – 11:00)

**TELA 1 (10:15 – 10:40):** `docs/relatorio_tecnico.md:367-381`, a seção 10, "Limitações".

**FALA:**
> "Sobre as limitações: as métricas medem anomalias injetadas, não acurácia clínica, que exigiria rótulos de especialistas. Os pesos da fusão são heurísticos, porque não existe dataset público com todas as modalidades do mesmo paciente, e as consultas de exemplo foram sintetizadas. E todos os indicadores são de triagem: apoiam a equipe, não substituem o julgamento clínico."

**TELA 2 (10:40 – 11:00):** `README.md`, topo, em preview, com o badge do CI verde visível.

**FALA:**
> "O código, o relatório técnico e o passo a passo para reproduzir estão no repositório, com mais de quatrocentos testes automatizados no CI. Esse foi o MedWatch, do Grupo 54. Obrigada!"

---

## Plano B (se algo falhar na hora)

| Problema | O que fazer |
|---|---|
| Azure falha no A5 | Rode `MEDWATCH_AZURE_MOCK=true medwatch audio analyze data/samples/consulta_02_dispneia.wav --patient PAC-001` e diga: "essas são respostas reais do Azure, gravadas no repositório". Mostre `docs/img/audio_azure_metricas_speech.png` |
| Dashboard não atualiza | Rode `medwatch simulate --scenario demo` no terminal: a saída lista os mesmos 14 alertas e termina com ✅ |
| Discord sem mensagens | Mostre `docs/img/alerta_webhook_discord.png` |
| Passou de 12 min | Corte a TELA 2 do B3 (noisy-OR) e a TELA 3 do A4 (relatório HTML) |
| Ficou abaixo de 10 min | Na Parte 2, mostre também `docs/relatorio_tecnico.md:265-271` (F1 da movimentação) ou abra a linha do tempo de um paciente no dashboard |

## Depois de gravar

- Subir no YouTube como **não listado**.
- Colar o link no `README.md` (linha 10) e no `docs/relatorio_tecnico.md` (linha 9).
- Gerar o PDF do relatório.
