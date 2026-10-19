# ADR 0001 — Late fusion orientada a eventos

- **Status:** aceito
- **Data:** 2026-10-17

## Contexto

O sistema combina sinais de naturezas diferentes: vídeo (pose e objetos), áudio (transcrição, sentimento, features vocais), séries temporais de sinais vitais, prescrições e movimentação. Cada modalidade tem taxa de amostragem, latência e ferramentas próprias, e os módulos são desenvolvidos em paralelo por pessoas diferentes do grupo.

## Decisão

Adotar **late fusion orientada a eventos**:

1. Cada pipeline processa sua modalidade de forma independente e publica `ClinicalEvent`s padronizados (`patient_id`, `source`, `type`, `score` ∈ [0, 1], `severity`, `evidence`, `artifacts`) em um `EventBus`.
2. Consumidores assinam o barramento. No PR 2 o `AlertManager` gera um alerta por evento acima da severidade mínima; no PR 7 o `RiskEngine` passa a fundir os eventos por paciente numa janela deslizante (noisy-OR ponderado e regras cross-modal).
3. O `AlertManager` entrega os alertas em canais plugáveis (console, JSONL, webhook), com deduplicação por paciente+tipo, cooldown e escalonamento quando a severidade sobe.

## Alternativas consideradas

- **Early fusion** (concatenar features de todas as modalidades em um único modelo): exige dados sincronizados e rotulados das três modalidades para o mesmo paciente, que não existem nos datasets públicos disponíveis, e acopla o desenvolvimento dos módulos.
- **Orquestração síncrona** (um pipeline chamando os demais em sequência): simples, mas uma modalidade lenta (vídeo) atrasaria alertas urgentes de outra (vitais).

## Consequências

- Os módulos podem ser desenvolvidos e testados isoladamente; o contrato `ClinicalEvent` é o único acoplamento.
- A fusão é explicável: cada `Alert` carrega em `reasons` os eventos que o motivaram.
- O `InMemoryEventBus` serve à execução local. Para produção, um adapter de Azure Event Hubs implementa o mesmo protocolo `EventBus` sem alterar produtores e consumidores.
- Interações sutis entre modalidades que só um modelo conjunto aprenderia ficam de fora; são parcialmente cobertas pelas regras cross-modal do RiskEngine.
