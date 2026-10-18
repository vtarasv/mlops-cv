"""Ingest + profile with an object-store subset, against the MLflow stack's S3-compatible store.

Runs in a scratch bucket of its own (created and deleted here), never the MLflow artifact one.
Requires the MLflow stack up and the S3 extra, which the lock leaves out: ``make test-all``
adds it.
"""

from __future__ import annotations

import json
import logging
import socket
import uuid
from pathlib import Path

import pytest

beam = pytest.importorskip("apache_beam")
boto3 = pytest.importorskip("boto3", reason="needs apache-beam[aws]: run `make test-all`")

# The atexit Prism shutdown warning crashes pytest's closed capture streams (see the unit tests).
logging.getLogger("apache_beam.utils.subprocess_server").setLevel(logging.ERROR)

from PIL import Image  # noqa: E402

from mlops_cv.data.subset import MANIFEST_FILENAME, RAW_SPLIT_DIRS  # noqa: E402
from mlops_cv.pipelines import ingest_pipeline, profile_pipeline, provenance  # noqa: E402
from mlops_cv.pipelines.profiling import PROFILE_DIRNAME, PROFILE_JSON, STAMP_FILENAME  # noqa: E402

pytestmark = pytest.mark.docker

COMPOSE_ENV = Path(__file__).resolve().parents[2] / "docker-compose" / ".env.mlflow"
STALE_KEY = "subset/images/train/STALE.jpg"


def _stack_env() -> dict[str, str]:
    text = COMPOSE_ENV.read_text(encoding="utf-8")
    return dict(
        line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#")
    )


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch):
    """A scratch bucket on the stack's store, with the store's standard env vars exported."""
    env = _stack_env()
    port = int(env["S3_API_PORT"])
    try:
        socket.create_connection(("localhost", port), timeout=2).close()
    except OSError as exc:
        pytest.skip(f"object store not reachable on localhost:{port}: {exc}")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", env["AWS_ACCESS_KEY_ID"])
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", env["AWS_SECRET_ACCESS_KEY"])
    monkeypatch.setenv("AWS_DEFAULT_REGION", env["AWS_DEFAULT_REGION"])
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", f"http://localhost:{port}")
    s3 = boto3.client("s3")
    name = f"pipelines-it-{uuid.uuid4().hex[:8]}"
    s3.create_bucket(Bucket=name)
    try:
        yield s3, name
    finally:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=name):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=name, Key=obj["Key"])
        s3.delete_bucket(Bucket=name)


def _raw_tree(root: Path) -> Path:
    """Two boxed frames per split; the test sequence doubles as the demo clip."""
    for split, seq in (("train", "seqA"), ("val", "seqB"), ("test", "seqC")):
        seq_dir = root / RAW_SPLIT_DIRS[split] / "sequences" / seq
        seq_dir.mkdir(parents=True)
        for idx in (1, 2):
            Image.new("RGB", (32, 24), (40 * idx, 90, 60)).save(seq_dir / f"{idx:07d}.jpg")
        ann = root / RAW_SPLIT_DIRS[split] / "annotations" / f"{seq}.txt"
        ann.parent.mkdir(parents=True)
        ann.write_text("1,0,4,4,8,8,1,1,0,0\n2,0,4,4,8,8,1,4,0,0\n", encoding="utf-8")
    return root


def _ingest(raw: Path, output_dir: str, demo_yaml: Path) -> None:
    argv = [f"--raw-dir={raw}", f"--output-dir={output_dir}", f"--demo-clips={demo_yaml}"]
    assert ingest_pipeline.main([*argv, "--frame-stride=1", "--runner=DirectRunner"]) == 0


def _keys(s3, bucket: str, prefix: str) -> set[str]:  # noqa: ANN001 - boto3 client
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return {obj["Key"][len(prefix) :] for page in pages for obj in page.get("Contents", [])}


def _body(s3, bucket: str, key: str) -> bytes:  # noqa: ANN001 - boto3 client
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def test_ingest_and_profile_read_and_write_an_object_store_subset(bucket, tmp_path: Path) -> None:
    s3, name = bucket
    raw = _raw_tree(tmp_path / "raw")
    demo_yaml = tmp_path / "demo_clips.yaml"
    demo_yaml.write_text("split: test\nfps: 30\nmax_side: 1920\nclips:\n  - seqC\n", "utf-8")
    local = tmp_path / "subset"
    remote = f"s3://{name}/subset"

    _ingest(raw, str(local), demo_yaml)
    _ingest(raw, remote, demo_yaml)
    s3.put_object(Bucket=name, Key=STALE_KEY, Body=b"stale")
    _ingest(raw, remote, demo_yaml)  # the re-ingest must reset the remote tree

    local_files = {str(p.relative_to(local)) for p in local.rglob("*") if p.is_file()}
    assert _keys(s3, name, "subset/") == local_files  # the stale frame is gone, nothing else is
    remote_manifest = _body(s3, name, f"subset/{MANIFEST_FILENAME}")
    local_manifest = (local / MANIFEST_FILENAME).read_bytes()
    assert sorted(remote_manifest.splitlines()) == sorted(local_manifest.splitlines())
    frame = "images/train/seqA_0000001.jpg"
    assert _body(s3, name, f"subset/{frame}") == (local / frame).read_bytes()

    staged: list[tuple[str, set[str]]] = []

    def publish(profile_dir: Path, stamp: str) -> str:
        staged.append((profile_dir.parent.name, {p.name for p in profile_dir.iterdir()}))
        return "data-run-1"

    assert profile_pipeline.main([f"--input-dir={remote}", "--runner=DirectRunner"], publish) == 0

    profile_keys = _keys(s3, name, f"subset/{PROFILE_DIRNAME}/")
    assert {PROFILE_JSON, STAMP_FILENAME} <= profile_keys
    # Publishing got a local copy of the profile, under the subset's name, before the stamp.
    assert staged == [("subset", profile_keys - {STAMP_FILENAME})]
    manifest_copy = tmp_path / MANIFEST_FILENAME
    manifest_copy.write_bytes(remote_manifest)
    profile = json.loads(_body(s3, name, f"subset/{PROFILE_DIRNAME}/{PROFILE_JSON}"))
    assert profile["source_manifest_sha256"] == provenance.manifest_sha256(manifest_copy)
