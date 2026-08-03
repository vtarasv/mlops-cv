"""Artifact hygiene: what a published NCNN model is allowed to contain."""

from __future__ import annotations

from pathlib import Path

from mlops_cv.optimize.artifacts import prune_ncnn_dir, size_mb


def _exported(tmp_path: Path) -> Path:
    """An NCNN export directory as the conversion leaves it, junk included."""
    model_dir = tmp_path / "ncnn-fp16-320_ncnn_model"
    (model_dir / "__pycache__").mkdir(parents=True)
    (model_dir / "model.ncnn.param").write_text("7767517\n", encoding="utf-8")
    (model_dir / "model.ncnn.bin").write_bytes(b"\x00" * 2048)
    (model_dir / "metadata.yaml").write_text("task: detect\n", encoding="utf-8")
    # The generated reference script bakes in the export-time temp directory.
    (model_dir / "model_ncnn.py").write_text(
        'net.load_param("/tmp/tmp7kfi4xfa/best_ncnn_model/model.ncnn.param")\n', encoding="utf-8"
    )
    # Bytecode of an intermediate module the export library deletes afterwards.
    (model_dir / "__pycache__" / "model_pnnx.cpython-312.pyc").write_bytes(b"\xcb\r\r\n")
    return model_dir


def test_pruning_keeps_exactly_what_the_runtime_loads(tmp_path: Path) -> None:
    """The loader globs a *.param, derives the .bin, and reads metadata.yaml — nothing else."""
    model_dir = _exported(tmp_path)
    prune_ncnn_dir(model_dir)
    assert sorted(p.name for p in model_dir.iterdir()) == [
        "metadata.yaml",
        "model.ncnn.bin",
        "model.ncnn.param",
    ]


def test_pruning_reports_what_it_removed(tmp_path: Path) -> None:
    assert sorted(prune_ncnn_dir(_exported(tmp_path))) == ["__pycache__", "model_ncnn.py"]


def test_no_build_host_paths_survive_in_the_published_model(tmp_path: Path) -> None:
    """A published artifact must not carry the filesystem layout of the machine that built it."""
    model_dir = _exported(tmp_path)
    prune_ncnn_dir(model_dir)
    text = " ".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in model_dir.rglob("*") if p.is_file()
    )
    assert "/tmp/" not in text


def test_pruning_is_idempotent(tmp_path: Path) -> None:
    """A re-published or already-clean export must not error or lose files."""
    model_dir = _exported(tmp_path)
    prune_ncnn_dir(model_dir)
    assert prune_ncnn_dir(model_dir) == []
    assert (model_dir / "model.ncnn.param").exists()


def test_directory_size_counts_every_file(tmp_path: Path) -> None:
    """NCNN artifacts are directories; the report's size column must not read as 0 MB."""
    assert size_mb(_exported(tmp_path)) > 0.002
