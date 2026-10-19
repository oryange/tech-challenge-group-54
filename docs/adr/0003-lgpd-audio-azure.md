# ADR 0003 — O que do áudio vai para o Azure e o que fica local (LGPD)

- **Status:** proposto
- **Data:** 2026-10-19

> Numerado como 0003 porque o ADR 0002 (backends de pose) está no PR 5, ainda aberto.

## Contexto

O enunciado exige Azure Speech to Text e Azure Text Analytics na análise das consultas. A voz e o conteúdo da consulta são **dados pessoais sensíveis referentes à saúde** (LGPD, art. 5º, II, e art. 11): a voz identifica a pessoa, e a transcrição traz sintomas, medicamentos e, às vezes, nomes. Enviar esses dados a um serviço de nuvem é tratamento por operador e precisa ser mínimo e justificado.

## Decisão

| Dado | Onde é processado | O que sai da máquina |
|---|---|---|
| Áudio da consulta | Azure Speech (`brazilsouth`) | O áudio já convertido em PCM 16 kHz mono, por stream e sem arquivo intermediário |
| Transcrição | Azure AI Language (mesma região do recurso) | O texto, dividido em documentos de até 5.000 caracteres, para Sentiment, frases-chave, resumo e Text Analytics for Health |
| Resumo e frases-chave | Devolvidos pelo Azure; guardados só localmente | Nada além do relatório local: são trechos da consulta e **não** entram em eventos, alertas nem webhook |
| F0, jitter, shimmer, HNR, pausas | Local (parselmouth/librosa) | Nada |
| Termos críticos e negação | Local (dicionário) | Nada |
| `ClinicalEvent` / alertas | Local; webhook opcional | Só metadados: categoria, termos do dicionário, contagens, z-scores. **Nenhum trecho da transcrição** |
| Relatório `outputs/reports/audio_<id>.json` | Local, fora do git | Nada (contém a transcrição completa) |
| Logs | Local | Duração, contagens e códigos de erro. Nunca o texto, o áudio ou as chaves |

Regras complementares:

1. **Pseudonimização:** o `patient_id` aceita só `[A-Za-z0-9_-]{1,64}` (ex.: `PAC-001`). Nome completo é recusado na CLI.
2. **Região:** os recursos ficam em `brazilsouth` (`scripts/provision_azure.sh`), sem transferência internacional.
3. **Retenção no Azure** (documentação de privacidade da Microsoft, consultada em 19/10/2026):
   - Speech to Text em tempo real processa o áudio só em memória e não guarda nada. Não usamos batch transcription nem Custom Speech, que guardam dados.
   - O Language pode guardar o texto por até 48 h para depuração. O opt-out (`LoggingOptOut`) vem ligado por padrão no Health, mas **desligado no Sentiment**. Por isso todas as chamadas passam `disable_service_logs=True`.
4. **Credenciais:** as chaves ficam só no `.env` (fora do git, barrado pelo hook `detect-private-key`) e são `SecretStr` no código. O endpoint do Language precisa ser HTTPS em domínio do Azure AI Services, para a chave não ser enviada a outro host.
5. **Respostas gravadas** (`data/samples/azure/`, usadas no modo mock): só de consultas **fictícias**. Esse diretório é versionado; uma consulta real gravada ali iria para o GitHub.
6. **Sem subprocess:** o áudio é decodificado pelo libsndfile dentro do processo. O caminho do arquivo nunca vira argumento de linha de comando.

## Alternativas consideradas

- **Whisper local** no lugar do Azure Speech: nada sairia da máquina, mas o enunciado exige Azure Speech to Text.
- **Mandar só trechos com termos críticos ao Language:** reduz o volume, mas o filtro prévio dependeria do mesmo dicionário que o Text Analytics for Health deveria complementar, e o sentimento perderia o contexto.
- **Anonimizar a transcrição antes do Language** (PII detection do próprio Azure): exigiria enviar o texto inteiro ao Azure do mesmo jeito. Fica como trabalho futuro, junto com a remoção de nomes no relatório local.

## Consequências

- Os indicadores acústicos não dependem da nuvem e continuam funcionando sem o Azure.
- O modo mock (`MEDWATCH_AZURE_MOCK=true`) permite CI e demonstração offline sem enviar nada.
- O relatório local é dado de saúde: não deve ser anexado a issues, PRs ou mensagens.
