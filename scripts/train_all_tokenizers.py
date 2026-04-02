#!/usr/bin/env python3
"""Train multiple SentencePiece tokenizers for the Bengali tokenizer study.

Trains tokenizers across two axes:
  - Algorithm: BPE vs Unigram
  - Vocab size: 16K, 32K, 49K, 64K
  - Training data: Bengali-only vs Bengali+English (50/50)

Outputs .model files to trained_tokenizers/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
import time

import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42
BYTES_PER_GB = 1 << 30

SPECIAL_TOKENS = [
    "<|user|>", "<|assistant|>", "<|end|>", "<|system|>",
    "<|pad|>", "<|sep|>", "<|cls|>", "<|mask|>",
]

CONFIGS = [
    # (name, algorithm, vocab_size, data_mix)
    # Bengali-only
    ("bn-bpe-16k", "bpe", 16000, "bn"),
    ("bn-bpe-32k", "bpe", 32000, "bn"),
    ("bn-bpe-49k", "bpe", 49152, "bn"),
    ("bn-bpe-64k", "bpe", 64000, "bn"),
    ("bn-uni-16k", "unigram", 16000, "bn"),
    ("bn-uni-32k", "unigram", 32000, "bn"),
    ("bn-uni-49k", "unigram", 49152, "bn"),
    ("bn-uni-64k", "unigram", 64000, "bn"),
    # Bengali + English mixed
    ("mix-bpe-32k", "bpe", 32000, "mix"),
    ("mix-bpe-49k", "bpe", 49152, "mix"),
    ("mix-bpe-64k", "bpe", 64000, "mix"),
    ("mix-uni-32k", "unigram", 32000, "mix"),
    ("mix-uni-49k", "unigram", 49152, "mix"),
]


def sample_jsonl(path: str, max_bytes: int, seed: int = SEED) -> list[str]:
    """Sample documents from a JSONL file up to max_bytes of UTF-8 text."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if text:
                texts.append(text)

    rng = random.Random(seed)
    rng.shuffle(texts)

    sampled = []
    total = 0
    for t in texts:
        sampled.append(t)
        total += len(t.encode("utf-8"))
        if total >= max_bytes:
            break

    log.info("Sampled %d docs (%.2f GB) from %s", len(sampled), total / BYTES_PER_GB, path)
    return sampled

def make_temp_sample(texts: list[str], prefix: str) -> str:
    """Write sampled texts to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        prefix=prefix,
        delete=False,
    ) as handle:
        for text in texts:
            handle.write(text.replace("\n", " ") + "\n")
        return handle.name


def train_one(
    name: str,
    algo: str,
    vocab_size: int,
    sample_path: str,
    output_dir: str,
) -> dict:
    """Train a single tokenizer and return metadata."""
    prefix = os.path.join(output_dir, name)

    log.info("Training: %s (algo=%s, vocab=%d)", name, algo, vocab_size)
    t0 = time.time()

    spm.SentencePieceTrainer.train(
        input=sample_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=algo,
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        split_by_whitespace=True,
        split_digits=True,
        num_threads=8,
        train_extremely_large_corpus=True,
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        user_defined_symbols=SPECIAL_TOKENS,
    )

    elapsed = time.time() - t0
    model_path = f"{prefix}.model"
    model_size = os.path.getsize(model_path)

    log.info("  Done in %.1fs, model size: %.1f MB", elapsed, model_size / 1e6)

    return {
        "name": name,
        "algorithm": algo,
        "vocab_size": vocab_size,
        "model_path": model_path,
        "model_size_bytes": model_size,
        "train_time_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bn-corpus", required=True, help="Bengali JSONL corpus")
    parser.add_argument("--en-corpus", default=None, help="English JSONL corpus (for mix configs)")
    parser.add_argument("--output-dir", required=True, help="Output directory for models")
    parser.add_argument("--sample-gb", type=float, default=1.5, help="GB to sample per language")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    sample_bytes = int(args.sample_gb * BYTES_PER_GB)

    # Sample Bengali data
    log.info("Sampling Bengali corpus...")
    bn_texts = sample_jsonl(args.bn_corpus, sample_bytes)

    # Sample English data if provided
    en_texts = []
    if args.en_corpus and os.path.exists(args.en_corpus):
        log.info("Sampling English corpus...")
        en_texts = sample_jsonl(args.en_corpus, sample_bytes)

    # Write sample files
    bn_sample = make_temp_sample(bn_texts, "bn_sample_")

    mix_sample = None
    if en_texts:
        # Interleave Bengali and English texts
        rng = random.Random(SEED)
        mixed = []
        for bn, en in zip(bn_texts, en_texts):
            mixed.append(bn)
            mixed.append(en)
        # Add remaining
        shorter = min(len(bn_texts), len(en_texts))
        mixed.extend(bn_texts[shorter:])
        mixed.extend(en_texts[shorter:])
        rng.shuffle(mixed)

        mix_sample = make_temp_sample(mixed, "mix_sample_")

    results = []
    for name, algo, vocab_size, data_mix in CONFIGS:
        if data_mix == "mix" and mix_sample is None:
            log.warning("Skipping %s: no English corpus provided", name)
            continue

        sample_path = mix_sample if data_mix == "mix" else bn_sample

        try:
            meta = train_one(name, algo, vocab_size, sample_path, output_dir)
            meta["data_mix"] = data_mix
            results.append(meta)
        except Exception as e:
            log.error("Failed to train %s: %s", name, e)

    # Save training metadata
    meta_path = os.path.join(output_dir, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Metadata saved to %s", meta_path)

    # Cleanup
    os.unlink(bn_sample)
    if mix_sample:
        os.unlink(mix_sample)

    log.info("All done. Trained %d tokenizers.", len(results))


if __name__ == "__main__":
    main()
