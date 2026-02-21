from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    extracted_dir: Path = data_dir / "extracted"
    outputs_dir: Path = project_root / "outputs"

def ensure_dirs(paths: Paths) -> None:
    paths.data_dir.mkdir(exist_ok=True)
    paths.raw_dir.mkdir(exist_ok=True, parents=True)
    paths.extracted_dir.mkdir(exist_ok=True, parents=True)
    paths.outputs_dir.mkdir(exist_ok=True, parents=True)
