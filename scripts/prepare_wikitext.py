#!/usr/bin/env python3
"""Download WikiText-103 and reproduce the deterministic GPT-2 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken


REVISION = "5fddba447aa4e75996922ea0d6b18b42f0a81cc4"
BASE_URL = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-103-raw-v1"
SHARDS = {
    "test": (("test-00000-of-00001.parquet", 732_610, "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),),
    "train": (
        ("train-00000-of-00002.parquet", 156_987_808, "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
        ("train-00001-of-00002.parquet", 157_088_770, "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"),
    ),
    "validation": (("validation-00000-of-00001.parquet", 657_209, "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"),),
}
EXPECTED_STREAMS = {
    "train": "567edc62db7882f2c69176215db23ba06c3b8443f4bf9363d9e0f89aa7fcb316",
    "validation": "05a6519598fb6bb77a8541933fd2a7cd69db8fd5ad33dadd57ce686dcf7752d4",
    "test": "56f9ff70776a35d689644535eb9b1b02eb3c3e7fb4df0c8a84d47b0fb5bde13f",
}
EXPECTED_MANIFEST = "aedbd8e0188c1377eb3313bba56c1f0e97e2e79583b60506b11f22067e2f154d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record(path: Path, *, dtype: str | None = None, shape: tuple[int, ...] | None = None) -> dict:
    value = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    if dtype is not None:
        value["dtype"] = dtype
    if shape is not None:
        value["shape"] = list(shape)
    return value


def download(url: str, path: Path, size: int, digest: str) -> None:
    if path.exists() and path.stat().st_size == size and sha256(path) == digest:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if temporary.stat().st_size != size or sha256(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {path.name}")
    temporary.replace(path)


def write_corpus(paths: list[Path], output: Path) -> None:
    with output.open("wb") as writer:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    encoded = (text or "").encode("utf-8")
                    writer.write(encoded)
                    if not encoded.endswith(b"\n"):
                        writer.write(b"\n")


def prepare_source(root: Path) -> Path:
    shard_root = root / "source"
    output = root / "wikitext103_raw_corpus"
    shard_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source_records: dict[str, list[dict]] = {}
    corpus_records: dict[str, dict] = {}
    for split, rows in SHARDS.items():
        paths = []
        for name, size, digest in rows:
            path = shard_root / name
            download(f"{BASE_URL}/{name}", path, size, digest)
            paths.append(path)
        source_records[split] = [record(path) for path in paths]
        corpus = output / f"wikitext103_{split}.utf8.bin"
        write_corpus(paths, corpus)
        corpus_records[split] = record(corpus, dtype="uint8", shape=(corpus.stat().st_size,))
    manifest = {
        "format": "wikitext103_raw_corpus_v1",
        "source_name": "WikiText-103 raw v1",
        "source_revision": REVISION,
        "source_shards": source_records,
        "corpora": corpus_records,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def tokenize(source: Path, output: Path, encoding: tiktoken.Encoding) -> int:
    count = 0
    pending = b""
    with source.open("rb") as reader, output.open("wb") as writer:
        while chunk := reader.read(8 << 20):
            pending += chunk
            boundary = pending.rfind(b"\n")
            if boundary < 0:
                continue
            complete, pending = pending[: boundary + 1], pending[boundary + 1 :]
            values = np.asarray(encoding.encode(complete.decode("utf-8"), disallowed_special=()), dtype=np.uint16)
            writer.write(values.tobytes(order="C"))
            count += len(values)
        if pending:
            values = np.asarray(encoding.encode(pending.decode("utf-8"), disallowed_special=()), dtype=np.uint16)
            writer.write(values.tobytes(order="C"))
            count += len(values)
    return count


def windows(corpus_path: Path, count: int, output: Path, shape: tuple[int, ...], seed: int) -> None:
    width = shape[-1]
    rows = int(np.prod(shape[:-1]))
    generator = np.random.default_rng(seed)
    offsets = generator.integers(0, count - width + 1, size=rows, dtype=np.int64)
    corpus = np.memmap(corpus_path, mode="r", dtype=np.uint16, shape=(count,))
    target = np.memmap(output, mode="w+", dtype=np.uint16, shape=(rows, width))
    positions = np.arange(width, dtype=np.int64)
    chunk_rows = max(1, (64 << 20) // (width * 2))
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        target[start:stop] = corpus[offsets[start:stop, None] + positions[None, :]]
    target.flush()


def prepare(root: Path) -> Path:
    cached = root / "wikitext103_gpt2_causal_2048/manifest.json"
    if cached.exists() and sha256(cached) == EXPECTED_MANIFEST:
        manifest = json.loads(cached.read_text())
        if all(
            sha256(cached.parent / manifest["records"][split]["path"])
            == EXPECTED_STREAMS[split]
            for split in EXPECTED_STREAMS
        ):
            print(cached)
            return cached
    source_manifest = prepare_source(root)
    source = json.loads(source_manifest.read_text())
    output = root / "wikitext103_gpt2_causal_2048"
    output.mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding("gpt2")
    token_records = {}
    token_counts = {}
    for split in ("train", "validation", "test"):
        corpus = source_manifest.parent / source["corpora"][split]["path"]
        path = output / f"wikitext103_gpt2_{split}_tokens.uint16.bin"
        token_counts[split] = tokenize(corpus, path, encoding)
        token_records[split] = record(path, dtype="uint16", shape=(token_counts[split],))
    specifications = (
        ("train", (4000, 8, 2049), 61_907),
        ("validation", (128, 2049), 10_061_907),
        ("test", (128, 2049), 20_061_907),
    )
    stream_records = {}
    for split, shape, seed in specifications:
        path = output / f"wikitext103_gpt2_{split}_stream.uint16.bin"
        windows(output / token_records[split]["path"], token_counts[split], path, shape, seed)
        stream_records[split] = record(path, dtype="uint16", shape=shape)
        if stream_records[split]["sha256"] != EXPECTED_STREAMS[split]:
            raise RuntimeError(f"generated {split} stream does not match the released experiment")
    manifest = {
        "format": "wikitext103_gpt2_causal_windows_v1",
        "source": "WikiText-103 raw v1",
        "source_url": "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/",
        "source_manifest": {"path": "../wikitext103_raw_corpus/manifest.json", "sha256": sha256(source_manifest)},
        "tokenization": "tiktoken:gpt2",
        "tokenizer_vocab_size": encoding.n_vocab,
        "storage_dtype": "uint16",
        "seq_len": 2048,
        "steps": 4000,
        "batch_size": 8,
        "eval_sequences": 128,
        "stream_seed": 61_907,
        "tokenized_corpora": token_records,
        "records": stream_records,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    if sha256(manifest_path) != EXPECTED_MANIFEST:
        raise RuntimeError("generated manifest does not match the released experiment")
    print(manifest_path)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/data"))
    prepare(parser.parse_args().output_root.resolve())
