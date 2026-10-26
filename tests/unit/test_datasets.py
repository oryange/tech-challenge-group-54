import hashlib

import httpx
import pytest

from medwatch.datasets import ChecksumError, PhysioNetDataset, download_dataset

DATASET = PhysioNetDataset(slug="demo", version="1.0", files=("icu/a.csv.gz",))
CONTENT = b"conteudo"


def transport(content: bytes, published: bytes = CONTENT, calls: list[str] | None = None):
    digest = hashlib.sha256(published).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path.endswith("SHA256SUMS.txt"):
            return httpx.Response(200, text=f"{digest} icu/a.csv.gz\n{digest} outro.txt\n")
        return httpx.Response(200, content=content)

    return httpx.MockTransport(handler)


def test_download_verifies_checksum(tmp_path):
    target = download_dataset(DATASET, tmp_path, transport=transport(CONTENT))
    assert target == tmp_path / "demo-1.0"
    assert (target / "icu" / "a.csv.gz").read_bytes() == CONTENT


def test_download_is_idempotent(tmp_path):
    download_dataset(DATASET, tmp_path, transport=transport(CONTENT))
    calls: list[str] = []
    download_dataset(DATASET, tmp_path, transport=transport(CONTENT, calls=calls))
    assert calls == ["/files/demo/1.0/SHA256SUMS.txt"]  # só confere, não baixa de novo


def test_corrupted_download_is_rejected_and_not_kept(tmp_path):
    with pytest.raises(ChecksumError):
        download_dataset(DATASET, tmp_path, transport=transport(b"corrompido"))
    folder = tmp_path / "demo-1.0" / "icu"
    assert list(folder.iterdir()) == []


def test_file_missing_from_checksums_is_rejected(tmp_path):
    dataset = PhysioNetDataset(slug="demo", version="1.0", files=("nao_existe.csv",))
    with pytest.raises(ChecksumError, match="não consta"):
        download_dataset(dataset, tmp_path, transport=transport(CONTENT))
