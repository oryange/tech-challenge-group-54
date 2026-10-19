import xml.etree.ElementTree as ET  # noqa: S405 - só analisa o SSML gerado pelo próprio teste

import pytest
from pydantic import ValidationError

from medwatch.audio.synthesis import Prosody, SamplesConfig, SampleScript, Turn, build_ssml
from tests.unit.audio.helpers import CONFIGS

VOICES = {"medico": "pt-BR-AntonioNeural", "paciente": "pt-BR-FranciscaNeural"}


def test_roteiros_do_repositorio_sao_validos():
    samples = SamplesConfig.load(CONFIGS / "audio_samples.yaml")
    assert {"consulta_01_normal", "consulta_02_dispneia"} <= set(samples.samples)
    for script in samples.samples.values():
        ET.fromstring(build_ssml(script, samples.voices))  # noqa: S314


def test_ssml_tem_vozes_prosodia_e_pausas():
    script = SampleScript(
        description="teste",
        patient_prosody=Prosody(rate="-30%", pitch="-10%"),
        phrase_break_ms=900,
        turn_break_ms=500,
        turns=[
            Turn(speaker="medico", phrases=["Como está?"]),
            Turn(speaker="paciente", phrases=["Cansada.", "Com falta de ar."]),
        ],
    )
    root = ET.fromstring(build_ssml(script, VOICES))  # noqa: S314
    ns = {"s": "http://www.w3.org/2001/10/synthesis"}
    voices = root.findall("s:voice", ns)
    assert [v.get("name") for v in voices] == list(VOICES.values())
    patient_prosody = voices[1].find("s:prosody", ns)
    assert patient_prosody.get("rate") == "-30%"
    assert voices[0].find("s:prosody", ns).get("rate") == "0%"
    assert [b.get("time") for b in patient_prosody.findall("s:break", ns)] == ["900ms"]


def test_texto_do_roteiro_e_escapado():
    script = SampleScript(
        description="teste",
        turns=[Turn(speaker="paciente", phrases=['Dor <forte> & "aguda"</prosody>'])],
    )
    root = ET.fromstring(build_ssml(script, VOICES))  # noqa: S314
    assert "".join(root.itertext()) == 'Dor <forte> & "aguda"</prosody>'


@pytest.mark.parametrize("rate", ["fast", "-300%", '0%" pitch="+50%'])
def test_prosodia_invalida_e_recusada(rate):
    with pytest.raises(ValidationError):
        Prosody(rate=rate)


def test_nome_de_amostra_e_voz_sao_validados():
    turn = {"speaker": "medico", "phrases": ["Oi."]}
    script = {"description": "x", "turns": [turn]}
    with pytest.raises(ValidationError):
        SamplesConfig.model_validate({"voices": VOICES, "samples": {"../fora": script}})
    with pytest.raises(ValidationError):
        SamplesConfig.model_validate(
            {"voices": {"medico": "voz<inválida>"}, "samples": {"ok": script}}
        )
