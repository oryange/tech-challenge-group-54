"""Download de datasets abertos do PhysioNet com verificação SHA-256 (idempotente)."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

PHYSIONET_BASE_URL = "https://physionet.org/files"


class ChecksumError(RuntimeError):
    """Arquivo baixado não confere com o SHA256SUMS publicado."""


@dataclass(frozen=True)
class PhysioNetDataset:
    slug: str
    version: str
    files: tuple[str, ...]

    @property
    def base_url(self) -> str:
        return f"{PHYSIONET_BASE_URL}/{self.slug}/{self.version}"

    @property
    def dirname(self) -> str:
        return f"{self.slug}-{self.version}"


# MIMIC-IV Clinical Database Demo: 100 pacientes, acesso aberto (ODbL), sem credenciamento.
MIMIC_IV_DEMO = PhysioNetDataset(
    slug="mimic-iv-demo",
    version="2.2",
    files=(
        "icu/chartevents.csv.gz",
        "icu/d_items.csv.gz",
        "icu/icustays.csv.gz",
        "hosp/patients.csv.gz",
        "hosp/prescriptions.csv.gz",
    ),
)

DATASETS = {MIMIC_IV_DEMO.slug: MIMIC_IV_DEMO}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checksums(client: httpx.Client, dataset: PhysioNetDataset) -> dict[str, str]:
    response = client.get(f"{dataset.base_url}/SHA256SUMS.txt")
    response.raise_for_status()
    checksums = {}
    for line in response.text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.strip()] = digest
    return checksums


def _fetch(client: httpx.Client, url: str, path: Path, expected: str) -> bool:
    """Baixa `url` em `path` se ainda não estiver íntegro. Retorna True se baixou."""
    if path.exists() and sha256(path) == expected:
        logger.info("{} já baixado e íntegro", path.name)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    logger.info("Baixando {} ...", path.name)
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with partial.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    if sha256(partial) != expected:
        partial.unlink()
        raise ChecksumError(f"Checksum inválido para {path.name}")
    partial.replace(path)
    return True


def _client(transport: httpx.BaseTransport | None) -> httpx.Client:
    # o Wikimedia exige User-Agent identificável com forma de contato; sem a URL, responde 429
    headers = {
        "User-Agent": (
            "medwatch-tech-challenge/0.1 "
            "(https://github.com/oryange/tech-challenge-group-54; academic project, POS Tech FIAP)"
        )
    }
    return httpx.Client(transport=transport, timeout=120, follow_redirects=True, headers=headers)


def download_file(
    url: str, path: Path, expected_sha256: str, transport: httpx.BaseTransport | None = None
) -> Path:
    with _client(transport) as client:
        _fetch(client, url, path, expected_sha256)
    return path


def download_dataset(
    dataset: PhysioNetDataset,
    data_dir: Path,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Baixa os arquivos para `data_dir/<slug>-<versão>`, pulando os que já conferem."""
    target_dir = data_dir / dataset.dirname
    with _client(transport) as client:
        checksums = fetch_checksums(client, dataset)
        for name in dataset.files:
            expected = checksums.get(name)
            if expected is None:
                raise ChecksumError(f"{name} não consta no SHA256SUMS de {dataset.dirname}")
            _fetch(client, f"{dataset.base_url}/{name}", target_dir / name, expected)
    return target_dir


@dataclass(frozen=True)
class VideoSample:
    filename: str
    url: str
    sha256: str
    description: str
    license: str
    credit: str
    crop: str | None = None  # "right": vídeo com profundidade|RGB lado a lado


# Vídeos de licença aberta para desenvolvimento e testes (não versionados no git).
VIDEO_SAMPLES: tuple[VideoSample, ...] = (
    VideoSample(
        filename="agachamento_unipodal.webm",
        url="https://upload.wikimedia.org/wikipedia/commons/1/16/Basic_single_leg_squat.webm",
        sha256="ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e",
        description="Agachamento unipodal, vista lateral (8 s)",
        license="CC BY-SA 4.0",
        credit="RickyBennison, Wikimedia Commons",
    ),
    VideoSample(
        filename="agachamento_barra.webm",
        url="https://upload.wikimedia.org/wikipedia/commons/5/5c/Squat_-_exercise_demonstration_video.webm",
        sha256="2440985661c3533a4ce78472b0f4577dbdf023aff3f8f9a225bbb5ff8071b1e9",
        description="Agachamento com barra, vista posterior (7 s)",
        license="CC BY 3.0",
        credit="FitnessScape, Wikimedia Commons",
    ),
    VideoSample(
        filename="urfall_queda_01.mp4",
        url="https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4",
        sha256="5fcb77e5b5a35d73ab8cda856ea58bd9e6acd7552e16fc6d22a623556bee76bc",
        description="Queda da posição em pé (UR Fall Detection Dataset, sequência fall-01)",
        license="Uso em pesquisa",
        credit="Kwolek & Kepski, University of Rzeszow (2014)",
        crop="right",
    ),
    VideoSample(
        filename="urfall_atividade_01.mp4",
        url="https://fenix.ur.edu.pl/~mkepski/ds/data/adl-01-cam0.mp4",
        sha256="b7b0220c8081a9a8b284174f55b7ca56e5bf9a525af1bfda0ae29c73d06016f8",
        description="Atividade normal: agachar e levantar (UR Fall Dataset, sequência adl-01)",
        license="Uso em pesquisa",
        credit="Kwolek & Kepski, University of Rzeszow (2014)",
        crop="right",
    ),
)
VIDEO_SAMPLES_BY_NAME = {sample.filename: sample for sample in VIDEO_SAMPLES}


def download_videos(videos_dir: Path, transport: httpx.BaseTransport | None = None) -> Path:
    with _client(transport) as client:
        for sample in VIDEO_SAMPLES:
            _fetch(client, sample.url, videos_dir / sample.filename, sample.sha256)
    return videos_dir


# UR Fall Detection Dataset (Kwolek & Kepski, 2014): câmera frontal (cam0), profundidade|RGB.
# Sequências 01–10 calibram a regra de queda; 11–20 são o conjunto de validação.
URFALL_BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
URFALL_SHA256: dict[str, str] = {
    "adl-01-cam0.mp4": "b7b0220c8081a9a8b284174f55b7ca56e5bf9a525af1bfda0ae29c73d06016f8",
    "adl-02-cam0.mp4": "46d63331b14aeafc0d9cb9ddac1f44f6040a801cad7ba6eba6b930f8a3a5994a",
    "adl-03-cam0.mp4": "44ec54c8ee227eaa4998fc5356bf8fd51a7d67d205ec44e3bfc3b4754fb3a3fa",
    "adl-04-cam0.mp4": "2694f06b1277d9de412c3f4d57bcef03f5396866dbfdd17fcc91718083bb8ac0",
    "adl-05-cam0.mp4": "01265dfff0a718f9b4adfa1a6aeed224d3cd5aa8141c050687703c27c59f607b",
    "adl-06-cam0.mp4": "a1022b9b4e5dbc2670ac5fbfa3854635c881f12a6a2529234fbf61a7fc635b8d",
    "adl-07-cam0.mp4": "0b346e78321c0cdfd281a19f3f4fd9cb6d61ed27f4caa1a955ac182c5b66a954",
    "adl-08-cam0.mp4": "85ae4a3c4539b1bcbb9542ae530f1c1b1ebc0c02124fa417d95f7944789dfa75",
    "adl-09-cam0.mp4": "6a54b645ff4207cbcb22043f3ab84c22e14261e237418f2cb9ea23b1c8f39672",
    "adl-10-cam0.mp4": "e8720d9d36f21c2e3222109b92d564ad3f1153927a9cf69dbd0eba0fcd003a23",
    "adl-11-cam0.mp4": "c798947d701b303ded1ed2b799a1423056564a71fe208639e517adc56ddfad7e",
    "adl-12-cam0.mp4": "77506f291304616dd5a1a7940ad43acbc5e1debccf675638eea3cea5d9481b42",
    "adl-13-cam0.mp4": "80a371d2cdbf5dcf277b7aa45539da6c04b778192aba45f0873a428b4d16727a",
    "adl-14-cam0.mp4": "d3d2f3e54b52da1d43b0445eb2f914cc1ea6922330dacb6f55ef65da3065b44e",
    "adl-15-cam0.mp4": "202d009146a01f8cd367e89dab06866082f59f12934a72898bbe450b65d61768",
    "adl-16-cam0.mp4": "9eb9d093c0fe04344591b2c2caf0d7d6204189d58a4a160b76d7eed7bf3b0f55",
    "adl-17-cam0.mp4": "ef2564f6680a4a2b790c67d562a77737ee005d8b0d8ab6c1ce92f83123100c93",
    "adl-18-cam0.mp4": "38710e818ebef54609801f1af69c513d23a78a72b5793247aa3e052965fd63f6",
    "adl-19-cam0.mp4": "4152673f37405d293ed911850742fadb367e7f0d8fafed28211b4f812802ebcd",
    "adl-20-cam0.mp4": "7524782dca9a9ee7a6cd215514bef372e132a3c9e7ad4786ea073560496ecc39",
    "fall-01-cam0.mp4": "5fcb77e5b5a35d73ab8cda856ea58bd9e6acd7552e16fc6d22a623556bee76bc",
    "fall-02-cam0.mp4": "4135a85724b5be8a248427cf49d4f28950cc1d2201434cf99f776fb4bbfd77b7",
    "fall-03-cam0.mp4": "6496581c9408ad87e7bbd75317e7112cb01b66755310990effa68a4badad2aff",
    "fall-04-cam0.mp4": "3cee6b1d27192184b299cce8fe289f013b1d69b89ba7bc92c025da849b10cece",
    "fall-05-cam0.mp4": "66ab179b17b12c240420ca5f9f95b73dbb922921d699009078dc61d8893c0c63",
    "fall-06-cam0.mp4": "66baa523914a199f1da70e476ba649a93c9046595b74606f854dc4267e596cca",
    "fall-07-cam0.mp4": "7dff29fad343856c2cc20c0bd1ab6b7ef749d75b916d75fc68cf0861793b39cd",
    "fall-08-cam0.mp4": "365d74ac9651b7dc7d61571614d4195d1517047ec0133330b5328ed0fe13055a",
    "fall-09-cam0.mp4": "28061e57e2daf23dd3cfc1a79d43800035035285b89170b7304a652d910f63e4",
    "fall-10-cam0.mp4": "2e077041e7f929ce120fc4c4f2c1e7421b9826c52d5d776c7341c67cf125095b",
    "fall-11-cam0.mp4": "d1274781ac2fb6f57d40b0337ab77573c882d06f86f5604073eb1568794382b4",
    "fall-12-cam0.mp4": "e3f463e492ff3d3119c318c4b24796c98e673a66d311c7eaa827be9c80b93e78",
    "fall-13-cam0.mp4": "172dee7c199de98348430713c5e8ad2c8be04ba38f7e6391c3c22047f968a024",
    "fall-14-cam0.mp4": "3c38f85993855f27d0549c369f89fde0449be50152cb9cdc06e6ab11f66fe10b",
    "fall-15-cam0.mp4": "85dd08909fb62efeb736ac159c595398521d8c70b7537663884dbd3b2dc583d9",
    "fall-16-cam0.mp4": "0ef51830d8573303eb2e5d2ea897f7bcf619843e570fc84aba0cd3d3c8c9a0b2",
    "fall-17-cam0.mp4": "ddb718f5064344abd56711d7ba2802e2e8a998d7433285a0732745681ffd9387",
    "fall-18-cam0.mp4": "5a6844e3644f472b3859448ddb9a4c8ca9dd761a623660a06c4220ee5267d412",
    "fall-19-cam0.mp4": "4013d44890d255c72fa981a9a43dbc449ae5940008341f7a907ecb233260e791",
    "fall-20-cam0.mp4": "01b27ea414b18f3a71e6d0f4e3cf1e1f9999dfd4a8af1b2c7dde74a5f0e0a5c3",
}


def urfall_split(split: str) -> list[str]:
    """Arquivos de um conjunto: "calibration" (01–10) ou "validation" (11–20)."""
    numbers = range(1, 11) if split == "calibration" else range(11, 21)
    return [f"{kind}-{i:02d}-cam0.mp4" for kind in ("fall", "adl") for i in numbers]


def download_urfall(target_dir: Path, transport: httpx.BaseTransport | None = None) -> Path:
    with _client(transport) as client:
        for name, digest in URFALL_SHA256.items():
            _fetch(client, f"{URFALL_BASE_URL}/{name}", target_dir / name, digest)
    return target_dir
