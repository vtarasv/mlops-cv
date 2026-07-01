"""Format evaluation metrics into MLflow artifacts: a markdown report + a metrics CSV."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlops_cv.eval.gate import GateResult

# Headline detection metrics, in display order, keyed by the ultralytics ``results.box`` attribute.
HEADLINE_ATTRS: dict[str, str] = {
    "precision": "mp",
    "recall": "mr",
    "mAP50": "map50",
    "mAP50-95": "map",
}


def headline_metrics(box: Any, prefix: str = "test") -> dict[str, float]:
    """Map an ultralytics ``results.box`` to ``{prefix/precision, recall, mAP50, mAP50-95}``.

    ``box`` is duck-typed (anything exposing ``mp``/``mr``/``map50``/``map``), so callers/tests can
    pass a stub and CI never imports ultralytics.
    """
    return {f"{prefix}/{name}": float(getattr(box, attr)) for name, attr in HEADLINE_ATTRS.items()}


def percentiles(samples: Sequence[float], ps: Sequence[float] = (50.0, 95.0)) -> dict[float, float]:
    """Linear-interpolation percentiles of ``samples`` (numpy's default method). Empty -> zeros."""
    if not samples:
        return {float(p): 0.0 for p in ps}
    ordered = sorted(float(s) for s in samples)
    n = len(ordered)
    out: dict[float, float] = {}
    for p in ps:
        rank = (p / 100.0) * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        out[float(p)] = ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])
    return out


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def comparison_table_md(
    candidate: Mapping[str, float],
    champion: Mapping[str, float] | None,
    *,
    metric_order: Sequence[str] = ("precision", "recall", "mAP50", "mAP50-95"),
    prefix: str = "test",
) -> str:
    """Markdown table of the headline metrics: ``Candidate`` (+ ``Champion``/``Δ`` if present)."""
    if champion is None:
        lines = ["| Metric | Candidate |", "|---|---|"]
        for name in metric_order:
            lines.append(f"| {name} | {_fmt(candidate.get(f'{prefix}/{name}'))} |")
        return "\n".join(lines)

    lines = ["| Metric | Candidate | Champion | Δ |", "|---|---|---|---|"]
    for name in metric_order:
        cand = candidate.get(f"{prefix}/{name}")
        champ = champion.get(f"{prefix}/{name}")
        delta = f"{cand - champ:+.4f}" if cand is not None and champ is not None else "—"
        lines.append(f"| {name} | {_fmt(cand)} | {_fmt(champ)} | {delta} |")
    return "\n".join(lines)


def _per_class_md(candidate: Mapping[str, float]) -> str:
    """A markdown table of per-class mAP50-95 (the ``.../mAP50-95/<class>`` keys), or ``""``."""
    rows = {k.rsplit("/", 1)[1]: v for k, v in candidate.items() if "/mAP50-95/" in k}
    if not rows:
        return ""
    lines = ["| Class | mAP50-95 |", "|---|---|"]
    lines += [f"| {name} | {_fmt(value)} |" for name, value in sorted(rows.items())]
    return "\n".join(lines)


def _latency_md(latency: Mapping[str, float]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in sorted(latency.items())]
    return "\n".join(lines)


def metrics_csv_rows(
    candidate: Mapping[str, float], champion: Mapping[str, float] | None
) -> list[dict[str, object]]:
    """Rows for ``csv.DictWriter``: the candidate's metrics (+ the champion's when present)."""
    rows: list[dict[str, object]] = [{"role": "candidate", **dict(candidate)}]
    if champion is not None:
        rows.append({"role": "champion", **dict(champion)})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["role"] + sorted({k for row in rows for k in row} - {"role"})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    out_dir: str | Path,
    *,
    candidate: Mapping[str, float],
    champion: Mapping[str, float] | None,
    gate: GateResult,
    latency: Mapping[str, float] | None = None,
    extra_md: str = "",
) -> tuple[Path, Path]:
    """Write ``report.md`` (gate verdict + tables) and ``metrics.csv``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sections = ["# Evaluation report", "", "## Gate", "", gate.summary(), ""]
    checks = list(getattr(gate, "checks", []))
    if checks:
        sections += ["| Check | Value | Threshold | Pass |", "|---|---|---|---|"]
        sections += [
            f"| {c.name} | {_fmt(c.value)} | {_fmt(c.threshold)} | {str(c.passed).upper()} |"
            for c in checks
        ]
        sections.append("")
    sections += ["## Metrics", "", comparison_table_md(candidate, champion), ""]

    per_class = _per_class_md(candidate)
    if per_class:
        sections += ["### Per class (mAP50-95)", "", per_class, ""]
    if latency:
        sections += ["## Latency", "", _latency_md(latency), ""]
    if extra_md:
        sections += [extra_md, ""]

    md_path = out / "report.md"
    md_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")

    csv_path = out / "metrics.csv"
    _write_csv(csv_path, metrics_csv_rows(candidate, champion))
    return md_path, csv_path
