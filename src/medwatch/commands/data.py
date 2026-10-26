"""medwatch data: download dos datasets abertos."""

import typer

from medwatch.config import get_settings
from medwatch.datasets import DATASETS, download_dataset, download_urfall, download_videos

app = typer.Typer(
    help="Datasets abertos (PhysioNet, vídeos de exemplo, UR Fall).", no_args_is_help=True
)

EXTRA_DATASETS = ("videos", "urfall")


@app.command()
def download(
    dataset: str = typer.Option(
        "mimic-iv-demo", help=f"Opções: {', '.join([*DATASETS, *EXTRA_DATASETS])}"
    ),
) -> None:
    """Baixa o dataset para data/raw/, conferindo o SHA-256 de cada arquivo."""
    raw = get_settings().data_dir / "raw"
    if dataset in DATASETS:
        target = download_dataset(DATASETS[dataset], raw)
    elif dataset == "videos":
        target = download_videos(raw / "videos")
    elif dataset == "urfall":
        target = download_urfall(raw / "urfall")
    else:
        options = ", ".join([*DATASETS, *EXTRA_DATASETS])
        typer.echo(f"Dataset desconhecido: {dataset}. Opções: {options}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"✅ {dataset} disponível em {target}")
