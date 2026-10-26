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


def download_dataset(
    dataset: PhysioNetDataset,
    data_dir: Path,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Baixa os arquivos para `data_dir/<slug>-<versão>`, pulando os que já conferem."""
    target_dir = data_dir / dataset.dirname
    with httpx.Client(transport=transport, timeout=60, follow_redirects=True) as client:
        checksums = fetch_checksums(client, dataset)
        for name in dataset.files:
            expected = checksums.get(name)
            if expected is None:
                raise ChecksumError(f"{name} não consta no SHA256SUMS de {dataset.dirname}")
            path = target_dir / name
            if path.exists() and sha256(path) == expected:
                logger.info("{} já baixado e íntegro", name)
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            logger.info("Baixando {} ...", name)
            with client.stream("GET", f"{dataset.base_url}/{name}") as response:
                response.raise_for_status()
                with partial.open("wb") as fh:
                    for chunk in response.iter_bytes():
                        fh.write(chunk)
            if sha256(partial) != expected:
                partial.unlink()
                raise ChecksumError(f"Checksum inválido para {name}")
            partial.replace(path)
    return target_dir
