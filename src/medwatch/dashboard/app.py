"""Dashboard da equipe médica (Streamlit): pacientes por risco, feed de alertas com ACK e
linha do tempo multimodal.

    streamlit run src/medwatch/dashboard/app.py      (ou `medwatch dashboard`)

Lê tudo da API (`MEDWATCH_API_URL`, padrão http://localhost:8000) e se atualiza sozinho.
"""

import os
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from medwatch.dashboard.client import (
    SEVERITY_COLORS,
    SEVERITY_ICONS,
    SEVERITY_ORDER,
    ApiError,
    MedWatchClient,
    contribution_rows,
    patient_rows,
    timeline_rows,
)

API_URL = os.environ.get("MEDWATCH_API_URL", "http://localhost:8000")
# horário de parede da equipe; a API trabalha sempre em UTC
TIMEZONE = os.environ.get("MEDWATCH_TIMEZONE", "America/Sao_Paulo")

st.set_page_config(page_title="MedWatch", page_icon="🩺", layout="wide")


@st.cache_resource
def client(url: str) -> MedWatchClient:
    return MedWatchClient(url)


def ack(api: MedWatchClient, alert_id: str) -> None:
    try:
        api.ack(alert_id)
        st.toast("Alerta reconhecido")
    except ApiError as exc:
        st.toast(f"Falha no ACK: {exc}")


def alert_card(api: MedWatchClient, alert: dict[str, Any]) -> None:
    severity = alert["severity"]
    created = pd.Timestamp(alert["created_at"]).tz_convert(TIMEZONE)
    with st.container(border=True):
        top, button = st.columns([5, 1])
        top.markdown(
            f"{SEVERITY_ICONS[severity]} **{severity.upper()}** · `{alert['patient_id']}` · "
            f"{created:%H:%M:%S}  \n**{alert['title']}**"
        )
        if alert["acked"]:
            button.caption("✅ reconhecido")
        else:
            button.button(
                "ACK", key=f"ack-{alert['alert_id']}", on_click=ack, args=(api, alert["alert_id"])
            )
        reasons = " · ".join(
            f"{r['source']}/{r['type']} ({r['score']:.2f})" for r in alert["reasons"]
        )
        st.caption(f"{alert['recommended_action']}  \nMotivos: {reasons}")


def timeline_chart(rows: list[dict[str, Any]]) -> alt.Chart:
    data = pd.DataFrame(rows)
    # horário de parede sem fuso: o Vega mostra datas sem fuso como estão, em qualquer navegador
    data["Horário"] = pd.to_datetime(data["Horário"]).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    severity = alt.Color(
        "Severidade:N",
        scale=alt.Scale(domain=SEVERITY_ORDER, range=SEVERITY_COLORS),
        sort=SEVERITY_ORDER,
        legend=alt.Legend(orient="top", title=None),
    )
    return (
        alt.Chart(data)
        .mark_circle(opacity=0.9, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("Horário:T", title=None, axis=alt.Axis(format="%H:%M")),
            y=alt.Y("Modalidade:N", title=None),
            color=severity,
            size=alt.Size("Score:Q", scale=alt.Scale(domain=[0, 1], range=[80, 500]), legend=None),
            tooltip=[
                alt.Tooltip("Horário:T", format="%H:%M:%S"),
                "Modalidade:N",
                "Evento:N",
                "Severidade:N",
                "Score:Q",
            ],
        )
        .properties(height=260)
    )


def contribution_chart(rows: list[dict[str, Any]]) -> alt.LayerChart:
    data = pd.DataFrame(rows).sort_values("Contribuição", ascending=False)
    order = data["Achado"].tolist()
    base = alt.Chart(data).encode(
        x=alt.X(
            "Contribuição:Q",
            scale=alt.Scale(domain=[0, 1.15]),
            axis=alt.Axis(values=[0, 0.5, 1], format=".1f"),
            title="peso × score",
        ),
        y=alt.Y("Achado:N", sort=order, title=None),
        tooltip=["Achado:N", alt.Tooltip("Contribuição:Q", format=".2f")],
    )
    bars = base.mark_bar(color="#2a78d6", cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=4, color="#9a9a96").encode(
        text=alt.Text("Contribuição:Q", format=".2f")
    )
    return (bars + labels).properties(height=alt.Step(28))


def patient_detail(api: MedWatchClient, patient_id: str) -> None:
    detail = api.patient(patient_id)
    if detail is None:
        st.info("Paciente sem eventos.")
        return
    severity = detail["severity"]
    st.markdown(
        f"#### {patient_id} · risco {detail['score']:.2f} "
        f"{SEVERITY_ICONS[severity]} {severity.upper()}"
    )
    left, right = st.columns([3, 2])
    with left:
        st.caption("Linha do tempo multimodal (tamanho = score)")
        st.altair_chart(timeline_chart(timeline_rows(detail)), use_container_width=True)
    with right:
        st.caption("Contribuições para o risco na janela atual (noisy-OR)")
        rows = contribution_rows(detail)
        if rows:
            st.altair_chart(contribution_chart(rows), use_container_width=True)
        else:
            st.info("Nenhum evento na janela de risco.")
    with st.expander("Eventos (tabela)"):
        st.dataframe(pd.DataFrame(timeline_rows(detail)), hide_index=True)


def main() -> None:
    api = client(API_URL)
    with st.sidebar:
        st.title("🩺 MedWatch")
        st.caption(f"API: {API_URL}")
        refresh = st.select_slider("Atualizar a cada (s)", options=[1, 2, 5, 10], value=2)
        only_open = st.toggle("Só alertas não reconhecidos", value=False)

    @st.fragment(run_every=refresh)
    def live() -> None:
        try:
            health = api.health()
            patients = api.patients()
            alerts = api.alerts(acked=False if only_open else None, limit=30)
        except ApiError as exc:
            st.error(f"API indisponível em {API_URL} ({exc}). Suba com `medwatch api`.")
            return

        open_alerts = [a for a in alerts if not a["acked"]]
        cols = st.columns(4)
        cols[0].metric("Pacientes monitorados", len(patients))
        cols[1].metric("Alertas abertos", sum(p["open_alerts"] for p in patients))
        cols[2].metric("Críticos abertos", sum(a["severity"] == "critical" for a in open_alerts))
        cols[3].metric("Eventos recebidos", health["events"])
        if not health["consumer_running"]:
            st.error("Consumidor de eventos parado: os alertas não estão sendo gerados.")

        ranking, feed = st.columns([3, 2])
        with ranking:
            st.subheader("Pacientes por risco")
            if patients:
                st.dataframe(
                    pd.DataFrame(patient_rows(patients)),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Risco": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.2f"
                        ),
                        "Último evento": st.column_config.DatetimeColumn(
                            format="HH:mm:ss", timezone=TIMEZONE
                        ),
                    },
                )
                ids = [p["patient_id"] for p in patients]
                selected = st.selectbox("Detalhar paciente", ids, key="patient")
                if selected:
                    patient_detail(api, selected)
            else:
                st.info("Nenhum evento recebido ainda. Rode `medwatch simulate --target api`.")
        with feed:
            st.subheader("Alertas em tempo real")
            for alert in alerts[:15]:
                alert_card(api, alert)

    live()


main()
