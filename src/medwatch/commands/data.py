"""medwatch data: download dos datasets abertos."""

import typer

from medwatch.config import get_settings
from medwatch.datasets import DATASETS, download_dataset

app = typer.Typer(help="Datasets abertos (PhysioNet).", no_args_is_help=True)


@app.command()
def download(
    dataset: str = typer.Option("mimic-iv-demo", help=f"Opções: {', '.join(DATASETS)}"),
) -> None:
    """Baixa o dataset para data/raw/, conferindo o SHA-256 de cada arquivo."""
    if dataset not in DATASETS:
        typer.echo(f"Dataset desconhecido: {dataset}. Opções: {', '.join(DATASETS)}", err=True)
        raise typer.Exit(code=1)
    target = download_dataset(DATASETS[dataset], get_settings().data_dir / "raw")
    typer.echo(f"✅ {dataset} disponível em {target}")
