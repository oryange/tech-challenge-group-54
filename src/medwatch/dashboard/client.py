"""Cliente HTTP da API usado pelo dashboard (separado do Streamlit para ser testável)."""

from typing import Any

import httpx

SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]
# rampa sequencial de um só matiz (laranja, claro → escuro): severidade é ordinal, e a
# legenda/tooltip sempre trazem o rótulo, então a cor nunca carrega o significado sozinha
SEVERITY_COLORS = ["#f6c3a8", "#f09a6e", "#eb6834", "#c24a1c", "#7f2a0b"]
SEVERITY_ICONS = {"info": "⚪", "low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}


class ApiError(RuntimeError):
    """API fora do ar ou resposta inesperada."""


class MedWatchClient:
    def __init__(
        self, base_url: str, timeout: float = 5.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _get(self, path: str, **params: Any) -> Any:
        try:
            response = self._client.get(
                path, params={k: v for k, v in params.items() if v is not None}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ApiError(f"{path}: {type(exc).__name__}") from exc
        return response.json()

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._get("/health")
        return result

    def patients(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._get("/patients")
        return result

    def patient(self, patient_id: str) -> dict[str, Any] | None:
        try:
            result: dict[str, Any] = self._get(f"/patients/{patient_id}/risk")
        except ApiError as exc:
            if (
                isinstance(exc.__cause__, httpx.HTTPStatusError)
                and exc.__cause__.response.status_code == 404
            ):
                return None
            raise
        return result

    def alerts(self, acked: bool | None = None, limit: int = 50) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._get("/alerts", acked=acked, limit=limit)
        return result

    def ack(self, alert_id: str) -> dict[str, Any]:
        try:
            response = self._client.post(f"/alerts/{alert_id}/ack")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ApiError(f"ack: {type(exc).__name__}") from exc
        result: dict[str, Any] = response.json()
        return result


def patient_rows(patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tabela "pacientes por risco"."""
    return [
        {
            "Paciente": p["patient_id"],
            "Risco": p["score"],
            "Severidade": f"{SEVERITY_ICONS[p['severity']]} {p['severity'].upper()}",
            "Modalidades": ", ".join(p["sources"]) or "—",
            "Alertas abertos": p["open_alerts"],
            "Último evento": p["last_event_at"],
        }
        for p in patients
    ]


def timeline_rows(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Pontos da linha do tempo multimodal de um paciente."""
    return [
        {
            "Horário": e["timestamp"],
            "Modalidade": e["source"],
            "Evento": e["type"],
            "Score": e["score"],
            "Severidade": e["severity"],
        }
        for e in detail["timeline"]
    ]


def contribution_rows(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Termos do noisy-OR: peso da modalidade × score do evento mais forte."""
    return [
        {
            "Achado": f"{c['source']}/{c['type']}",
            "Contribuição": round(c["weight"] * c["score"], 3),
        }
        for c in detail["contributions"]
    ]
