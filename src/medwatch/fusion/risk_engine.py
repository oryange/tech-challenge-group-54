"""Fusão multimodal: risco agregado por paciente e regras de escalonamento cross-modal.

O `RiskEngine` recebe os `ClinicalEvent`s de todos os módulos e, para cada um, devolve os
alertas que ele provoca:

1. **Direto**: o próprio evento vira alerta com a severidade dele (`Alert.from_event`).
2. **Regra cross-modal**: combinações de `configs/fusion.yaml` (ex.: falta de ar no áudio +
   queda de SpO2 nos vitais) geram um alerta `fusion.<regra>` com os eventos da combinação.
3. **Risco combinado**: o noisy-OR ponderado dos eventos na janela gera `fusion.risk_score`
   quando há pelo menos duas modalidades e a faixa de risco supera o pior evento individual.

O filtro de severidade mínima e o cooldown ficam no `AlertManager`.
"""

import bisect
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from medwatch.core.events import RECOMMENDED_ACTIONS, Alert, ClinicalEvent, Severity, Source
from medwatch.fusion.config import FusionConfig, FusionRule

RISK_ALERT_TYPE = "fusion.risk_score"


class Contribution(BaseModel):
    """Termo do noisy-OR: o evento mais forte de um (fonte, tipo) na janela."""

    model_config = ConfigDict(frozen=True)

    source: Source
    type: str
    score: float
    weight: float
    event_id: str

    @property
    def weighted(self) -> float:
        return self.weight * self.score


class PatientRisk(BaseModel):
    """Retrato do risco de um paciente no instante `as_of`."""

    model_config = ConfigDict(frozen=True)

    patient_id: str
    score: float
    severity: Severity
    sources: tuple[Source, ...]
    contributions: tuple[Contribution, ...]
    events: tuple[ClinicalEvent, ...]
    as_of: datetime


def noisy_or(contributions: Iterable[Contribution]) -> float:
    """R = 1 − Π(1 − wᵢ·scoreᵢ): cada termo é tratado como evidência independente."""
    survival = 1.0
    for item in contributions:
        survival *= 1.0 - item.weighted
    return 1.0 - survival


class RiskEngine:
    """Mantém o histórico recente de cada paciente e decide os alertas de cada evento.

    Os alertas dependem só do horário dos eventos: regras e risco combinado são avaliados no
    instante do evento que chegou, então um replay de dados históricos gera os mesmos alertas
    que o tempo real. O retrato (`risk`, `patients`) usa como "agora" o evento mais recente
    recebido ou o `clock()`, o que for maior: com relógio, o risco de um paciente sem eventos
    novos decai sozinho conforme a janela passa.
    """

    def __init__(self, config: FusionConfig, clock: Callable[[], datetime] | None = None) -> None:
        self.config = config
        self._clock = clock
        self._history: dict[str, list[ClinicalEvent]] = defaultdict(list)
        self._latest: datetime | None = None

    # --- estado ------------------------------------------------------------------------

    def now(self) -> datetime | None:
        if self._clock is None:
            return self._latest
        clock = self._clock()
        return clock if self._latest is None else max(clock, self._latest)

    def _remember(self, event: ClinicalEvent) -> None:
        history = self._history[event.patient_id]
        # eventos podem chegar fora de ordem (vídeo processado depois dos vitais)
        bisect.insort(history, event, key=lambda e: e.timestamp)
        if self._latest is None or event.timestamp > self._latest:
            self._latest = event.timestamp
        self._prune(event.patient_id)

    def _prune(self, patient_id: str) -> None:
        # pela linha do tempo do próprio paciente: um replay de dados antigos não apaga quem
        # está sendo monitorado ao vivo
        history = self._history[patient_id]
        cutoff = history[-1].timestamp - self.config.history
        keep_from = bisect.bisect_left(history, cutoff, key=lambda e: e.timestamp)
        del history[:keep_from]

    def window_events(self, patient_id: str, at: datetime | None = None) -> list[ClinicalEvent]:
        """Eventos do paciente na janela de risco que termina em `at` (padrão: agora)."""
        end = at or self.now()
        if end is None:
            return []
        start = end - self.config.window
        return [e for e in self._history.get(patient_id, []) if start <= e.timestamp <= end]

    # --- risco ---------------------------------------------------------------------------

    def contributions(self, events: Iterable[ClinicalEvent]) -> list[Contribution]:
        strongest: dict[tuple[Source, str], ClinicalEvent] = {}
        for event in events:
            key = (event.source, event.type)
            if key not in strongest or event.score > strongest[key].score:
                strongest[key] = event
        return [
            Contribution(
                source=e.source,
                type=e.type,
                score=e.score,
                weight=self.config.weight(e.source),
                event_id=str(e.event_id),
            )
            for e in sorted(strongest.values(), key=lambda e: e.timestamp)
            if self.config.weight(e.source) > 0
        ]

    def risk(self, patient_id: str, at: datetime | None = None) -> PatientRisk:
        """Risco na janela que termina em `at` (padrão: agora)."""
        events = self.window_events(patient_id, at)
        terms = self.contributions(events)
        score = noisy_or(terms)
        return PatientRisk(
            patient_id=patient_id,
            score=round(score, 4),
            severity=self.config.risk_bands.severity(score),
            sources=tuple(sorted({t.source for t in terms})),
            contributions=tuple(terms),
            events=tuple(events),
            as_of=at or self.now() or datetime.min.replace(tzinfo=UTC),
        )

    def patients(self) -> list[PatientRisk]:
        """Todos os pacientes com histórico, do maior risco para o menor."""
        risks = [self.risk(patient_id) for patient_id in self._history]
        return sorted(risks, key=lambda r: (-r.score, r.patient_id))

    # --- alertas -------------------------------------------------------------------------

    def alerts_for(self, event: ClinicalEvent) -> list[Alert]:
        """Registra o evento e devolve os alertas que ele provoca (sem filtro de severidade)."""
        self._remember(event)
        alerts = [Alert.from_event(event)]
        history = self._history[event.patient_id]
        for rule in self.config.rules:
            combination = match_rule(rule, event, history)
            if combination:
                alerts.append(
                    Alert(
                        patient_id=event.patient_id,
                        alert_type=f"fusion.{rule.name}",
                        severity=rule.severity,
                        title=rule.title,
                        reasons=combination,
                        recommended_action=RECOMMENDED_ACTIONS[rule.severity],
                    )
                )
        risk_alert = self._risk_alert(event)
        # uma regra que já disparou com severidade igual ou maior explica melhor o mesmo risco
        if risk_alert and all(a.severity < risk_alert.severity for a in alerts[1:]):
            alerts.append(risk_alert)
        return alerts

    def _risk_alert(self, trigger: ClinicalEvent) -> Alert | None:
        patient_id = trigger.patient_id
        risk = self.risk(patient_id, at=trigger.timestamp)
        if len(risk.sources) < self.config.min_sources or not risk.events:
            return None
        worst = max(e.severity for e in risk.events)
        # só escala quando a combinação diz mais que o pior evento sozinho
        if risk.severity <= worst:
            return None
        by_id = {str(e.event_id): e for e in risk.events}
        reasons = tuple(by_id[c.event_id] for c in risk.contributions)
        sources = " + ".join(s.value for s in risk.sources)
        return Alert(
            patient_id=patient_id,
            alert_type=RISK_ALERT_TYPE,
            severity=risk.severity,
            title=f"Risco combinado {risk.score:.2f} ({sources})",
            reasons=reasons,
            recommended_action=RECOMMENDED_ACTIONS[risk.severity],
        )


def match_rule(
    rule: FusionRule, trigger: ClinicalEvent, history: list[ClinicalEvent]
) -> tuple[ClinicalEvent, ...] | None:
    """Eventos que completam a regra junto com `trigger`, ou None.

    A regra só dispara no evento que a completa: um evento novo sem relação com a regra não
    a repete. Para cada outra condição escolhe o evento mais forte dentro da janela. Com
    `rule.ordered`, cada condição só aceita eventos depois da anterior e antes da seguinte.
    """
    slots = [i for i, condition in enumerate(rule.all_of) if condition.matches(trigger)]
    if not rule.ordered:
        slots = slots[:1]
    for slot in slots:
        events = _complete(rule, slot, trigger, history)
        if events:
            return events
    return None


def _complete(
    rule: FusionRule, slot: int, trigger: ClinicalEvent, history: list[ClinicalEvent]
) -> tuple[ClinicalEvent, ...] | None:
    """Preenche as outras condições com o `trigger` ocupando a condição `slot`."""
    chosen: dict[int, ClinicalEvent] = {slot: trigger}
    used = {str(trigger.event_id)}
    # na regra ordenada, parte do trigger para os dois lados: cada escolha limita a próxima
    others = [*range(slot - 1, -1, -1), *range(slot + 1, len(rule.all_of))]
    for index in others:
        condition = rule.all_of[index]
        candidates = [
            e
            for e in history
            if condition.matches(e)
            and str(e.event_id) not in used
            and abs(e.timestamp - trigger.timestamp) <= rule.window
        ]
        if rule.ordered:
            if index < slot:
                candidates = [e for e in candidates if e.timestamp <= chosen[index + 1].timestamp]
            else:
                candidates = [e for e in candidates if e.timestamp >= chosen[index - 1].timestamp]
        if not candidates:
            return None
        best = max(candidates, key=lambda e: (e.score, e.timestamp))
        chosen[index] = best
        used.add(str(best.event_id))
    events = sorted(chosen.values(), key=lambda e: e.timestamp)
    if events[-1].timestamp - events[0].timestamp > rule.window:
        return None
    return tuple(events)
