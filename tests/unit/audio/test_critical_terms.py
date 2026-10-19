import pytest

from medwatch.audio.azure_language import HealthEntity
from medwatch.audio.critical_terms import CriticalTermDetector, normalize, tokenize
from tests.unit.audio.helpers import terms_config


@pytest.fixture(scope="module")
def detector():
    return CriticalTermDetector(terms_config())


def statuses(detector, text, category="dyspnea"):
    return [m.status for m in detector.detect(text) if m.category == category]


def test_normalizacao_preserva_offsets():
    text = "Sinto CANSAÇO e Falta de Ar"
    assert normalize(text) == "sinto cansaco e falta de ar"
    assert len(normalize(text)) == len(text)
    assert [t.text for t in tokenize(text)][:2] == ["sinto", "cansaco"]


def test_termo_afirmado(detector):
    mentions = detector.detect("Estou com falta de ar desde ontem.")
    assert [(m.category, m.term, m.status) for m in mentions] == [
        ("dyspnea", "falta de ar", "affirmed")
    ]
    text = "Estou com falta de ar desde ontem."
    assert text[mentions[0].start : mentions[0].end] == "falta de ar"


def test_nao_tenho_falta_de_ar_nao_dispara(detector):
    assert statuses(detector, "Não tenho falta de ar.") == ["negated"]


@pytest.mark.parametrize(
    "text",
    [
        "Nunca senti falta de ar.",
        "Paciente nega falta de ar.",
        "Estou sem falta de ar hoje.",
        "Falta de ar, não.",
        "Falta de ar nenhuma",
    ],
)
def test_outras_formas_de_negacao(detector, text):
    assert statuses(detector, text) == ["negated"]


def test_negacao_nao_atravessa_mas(detector):
    text = "Não tenho febre, mas estou com falta de ar."
    assert statuses(detector, text) == ["affirmed"]


def test_negacao_nao_atravessa_frase(detector):
    assert statuses(detector, "Não dormi. Acordei com falta de ar.") == ["affirmed"]


def test_negacao_tem_alcance_limitado(detector):
    text = "Não fui ao médico porque estava com falta de ar."
    assert statuses(detector, text) == ["affirmed"]


def test_nao_como_resposta_nao_nega(detector):
    assert statuses(detector, "Não, estou com falta de ar.") == ["affirmed"]


def test_negacao_posposta_seguida_de_verbo_nao_nega(detector):
    text = "A dor no peito não melhora."
    assert statuses(detector, text, "chest_pain") == ["affirmed"]


def test_palavra_de_outro_sintoma_nao_nega(detector):
    # o "sem" de "sem ar" é parte do sintoma, não negação de "cansada"
    mentions = {m.term: m.status for m in detector.detect("Fico sem ar e cansada.")}
    assert mentions == {"sem ar": "affirmed", "cansada": "affirmed"}


def test_negacao_alcanca_sintomas_coordenados(detector):
    mentions = {m.term: m.status for m in detector.detect("Não tenho falta de ar e cansaço.")}
    assert mentions == {"falta de ar": "negated", "cansaco": "negated"}


def test_pergunta_nao_conta_como_sintoma(detector):
    assert statuses(detector, "Sentiu falta de ar essa noite?") == ["question"]


def test_termo_que_contem_nao(detector):
    assert statuses(detector, "Eu não consigo respirar direito.") == ["affirmed"]


def test_termo_de_varias_palavras_nao_atravessa_pontuacao(detector):
    assert statuses(detector, "Não, consigo respirar bem.") == []


def test_acentos_e_maiusculas(detector):
    assert statuses(detector, "Sinto muito CANSAÇO.", "fatigue") == ["affirmed"]


def test_negacao_do_text_analytics_for_health(detector):
    text = "Relata tontura ao levantar."
    start = text.index("tontura")
    entity = HealthEntity(
        text="tontura",
        category="SymptomOrSign",
        confidence=0.9,
        offset=start,
        length=len("tontura"),
        certainty="negative",
    )
    [mention] = detector.detect(text, [entity])
    assert mention.status == "negated"
    assert mention.negated_by == "health"


def test_entidade_positiva_nao_nega(detector):
    text = "Relata tontura ao levantar."
    entity = HealthEntity(
        text="tontura",
        category="SymptomOrSign",
        confidence=0.9,
        offset=text.index("tontura"),
        length=7,
        certainty="positive",
    )
    assert [m.status for m in detector.detect(text, [entity])] == ["affirmed"]
