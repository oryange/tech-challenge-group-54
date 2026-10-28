# Roteiro de gravação — consultas de áudio (voz real)

As amostras sintéticas (`medwatch audio synth-samples`) garantem testes reprodutíveis, mas a voz neural não tem as alterações reais de fadiga e dispneia (jitter, shimmer, soprosidade, respiração audível). Para a demonstração, duas pessoas do grupo gravam as mesmas consultas com voz real.

## Como gravar

- Ambiente silencioso, celular a cerca de 30 cm, sem música nem ventilador.
- Formato aceito: `.wav`, `.flac`, `.ogg` (inclusive áudio do WhatsApp) ou `.mp3`. Gravação do iPhone (`.m4a`) precisa ser convertida antes:
  `ffmpeg -i consulta.m4a -ac 1 -ar 16000 consulta.wav`
- Uma pessoa faz o médico, outra o paciente. Ninguém diz nomes: o paciente é "a senhora" / "o senhor".
- Duração esperada: 40 s a 1 min 30 s por consulta.

## Consulta 1 — normal

Paciente com fala no ritmo habitual, sem pausas longas.

> **Médico:** Bom dia. Como a senhora passou a noite?
> **Paciente:** Bom dia, doutor. Passei bem, dormi a noite toda.
> **Médico:** Sentiu alguma falta de ar ou cansaço para fazer as coisas?
> **Paciente:** Não, não tenho falta de ar. Ontem caminhei duas vezes pelo corredor e me senti disposta.
> **Médico:** E dor no peito ou tontura?
> **Paciente:** Nenhuma dor no peito. Também não tive tontura.
> **Médico:** Está se alimentando bem?
> **Paciente:** Estou, sim. Comi tudo no almoço e o apetite voltou.
> **Médico:** Ótimo. Vamos manter a medicação e reavaliar amanhã.
> **Paciente:** Combinado, doutor. Obrigada.

Esperado: nenhum termo crítico afirmado (os sintomas aparecem negados ou na pergunta do médico) e nenhum evento ≥ medium.

## Consulta 2 — dispneia e fadiga

Paciente fala devagar, em frases curtas, respirando entre elas (pausa audível de ~1 s), com voz mais baixa e cansada.

> **Médico:** Bom dia. Como a senhora passou a noite?
> **Paciente:** Bom dia, doutor. *(pausa)* Passei mal. *(pausa)* Acordei várias vezes com falta de ar. *(pausa)* Tive que dormir sentada.
> **Médico:** E durante o dia, como está?
> **Paciente:** Estou muito cansada. *(pausa)* Me canso só de ir até o banheiro. *(pausa)* Fico sem ar no meio do caminho.
> **Médico:** Está sentindo dor no peito?
> **Paciente:** Não sinto dor no peito. *(pausa)* É mais esse cansaço para respirar.
> **Médico:** Vou ajustar o oxigênio e pedir uma gasometria. Qualquer piora, chame a enfermagem.
> **Paciente:** Está bem. *(pausa)* Obrigada, doutor.

Esperado: evento `dyspnea_terms` (medium) e `fatigue_terms` (low); dor no peito registrada como negada; indicador de dificuldade respiratória elevado pelas pausas.

## Depois de gravar

```bash
medwatch audio analyze gravacoes/consulta_normal.wav --patient PAC-101
# usa a consulta normal como baseline do mesmo "paciente"
medwatch audio analyze gravacoes/consulta_dispneia.wav --patient PAC-101 \
  --baseline outputs/reports/audio_PAC-101_<data>.json --alert
```

As gravações com voz real **não vão para o repositório**: a voz identifica quem gravou. Guarde-as numa pasta fora do git (ex.: `gravacoes/`, que está no `.gitignore`).
