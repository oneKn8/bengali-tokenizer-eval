#!/usr/bin/env python3
"""Evaluate tokenizers on Bengali text across multiple metrics.

Compares both trained SentencePiece tokenizers and HuggingFace tokenizers
(Llama, Qwen, BanglaBERT, etc.) on the same held-out Bengali eval set.

Metrics:
  1. Fertility: avg tokens per whitespace-delimited word
  2. Sequence length: avg tokens per document (context efficiency)
  3. Unknown/byte-fallback rate: % of tokens that are UNK or byte-level
  4. Script purity: % of tokens containing only Bengali + Latin + punctuation
  5. Compression ratio: bytes per token (higher = more efficient)
  6. Morphological segmentation quality (qualitative, on probe words)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

# Bengali script range: U+0980-U+09FF
# Also allow Devanagari danda U+0964-0965 (standard Bengali punctuation)
_BENGALI_RANGE = set(range(0x0980, 0x0A00)) | {0x0964, 0x0965}
_LATIN_RANGE = set(range(0x0000, 0x0080)) | set(range(0x00A0, 0x0100))


def classify_token_script(token: str) -> str:
    """Classify a token's script content.

    Returns one of: 'bengali', 'latin', 'mixed', 'punctuation', 'numeric',
    'byte_fallback', 'special', 'other_indic', 'other'
    """
    # Strip SentencePiece whitespace marker
    clean = token.replace("\u2581", "").replace("_", "")
    if not clean:
        return "punctuation"

    # Byte fallback tokens: <0xAB>
    if re.match(r"^<0x[0-9A-Fa-f]{2}>$", token):
        return "byte_fallback"

    # Special tokens
    if token.startswith("<") and token.endswith(">"):
        return "special"

    has_bengali = False
    has_latin = False
    has_other_indic = False
    has_other = False
    all_punct_or_num = True

    for ch in clean:
        cp = ord(ch)
        cat = unicodedata.category(ch)

        if cat.startswith("P") or cat.startswith("S") or cat == "Zs":
            continue  # punctuation/symbol/space
        if cat.startswith("N"):
            continue  # numeric

        all_punct_or_num = False

        if cp in _BENGALI_RANGE:
            has_bengali = True
        elif cp in _LATIN_RANGE:
            has_latin = True
        elif 0x0900 <= cp <= 0x0D7F:  # Other Indic scripts
            has_other_indic = True
        else:
            has_other = True

    if all_punct_or_num:
        if any(unicodedata.category(ch).startswith("N") for ch in clean):
            return "numeric"
        return "punctuation"

    if has_other_indic:
        return "other_indic"
    if has_bengali and has_latin:
        return "mixed"
    if has_bengali:
        return "bengali"
    if has_latin:
        return "latin"
    return "other"


# ---------------------------------------------------------------------------
# Tokenizer wrappers
# ---------------------------------------------------------------------------

class SPMTokenizer:
    """Wrapper for SentencePiece models."""

    def __init__(self, model_path: str, name: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.name = name
        self.vocab_size = self.sp.get_piece_size()
        self.source = "sentencepiece"

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def encode_str(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def get_piece(self, id: int) -> str:
        return self.sp.id_to_piece(id)


class HFTokenizer:
    """Wrapper for HuggingFace tokenizers."""

    def __init__(self, model_id: str, name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.name = name
        self.vocab_size = self.tok.vocab_size
        self.source = "huggingface"

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def encode_str(self, text: str) -> list[str]:
        ids = self.encode(text)
        return [self.tok.decode([i]) for i in ids]

    def get_piece(self, id: int) -> str:
        return self.tok.decode([id])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_eval_texts(path: str, max_docs: int = 5000) -> list[str]:
    """Load evaluation texts from JSONL."""
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
            if len(text) >= 100:  # skip very short docs
                texts.append(text)
            if len(texts) >= max_docs:
                break
    return texts


def compute_fertility(tokenizer, texts: list[str]) -> float:
    """Average tokens per whitespace word."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.split()
        total_words += len(words)
        total_tokens += len(tokenizer.encode(text))
    return total_tokens / max(total_words, 1)


def compute_compression_ratio(tokenizer, texts: list[str]) -> float:
    """Average bytes per token (higher = more compression per token)."""
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))
    return total_bytes / max(total_tokens, 1)


def compute_avg_doc_tokens(tokenizer, texts: list[str]) -> float:
    """Average tokens per document."""
    total = sum(len(tokenizer.encode(t)) for t in texts)
    return total / max(len(texts), 1)


def compute_script_distribution(tokenizer, texts: list[str]) -> dict[str, float]:
    """Script distribution of tokens across the eval set."""
    counter: Counter = Counter()
    total = 0
    for text in texts[:1000]:  # sample for speed
        tokens = tokenizer.encode_str(text)
        for t in tokens:
            script = classify_token_script(t)
            counter[script] += 1
            total += 1

    return {k: round(v / max(total, 1) * 100, 2) for k, v in counter.most_common()}


def compute_byte_fallback_rate(tokenizer, texts: list[str]) -> float:
    """Percentage of tokens that are byte-fallback."""
    total = 0
    byte_tokens = 0
    for text in texts[:2000]:
        tokens = tokenizer.encode_str(text)
        total += len(tokens)
        for t in tokens:
            if re.match(r"^<0x[0-9A-Fa-f]{2}>$", t):
                byte_tokens += 1
    return byte_tokens / max(total, 1) * 100


# Morphological probe words: common Bengali words with rich morphology
PROBE_WORDS = [
    "বাংলাদেশ",           # Bangladesh
    "বাংলাদেশের",         # Bangladesh's (genitive)
    "বিশ্ববিদ্যালয়",      # university
    "বিশ্ববিদ্যালয়ে",     # at university (locative)
    "বিশ্ববিদ্যালয়গুলোতে", # at universities (plural + locative)
    "প্রধানমন্ত্রী",       # prime minister
    "জনগণ",              # people
    "জনগণের",            # people's (genitive)
    "আন্তর্জাতিক",       # international
    "গণপ্রজাতন্ত্রী",     # republic (compound)
    "স্বাধীনতা",          # independence
    "স্বাধীনতার",         # of independence
    "কম্পিউটার",         # computer (loanword)
    "ইন্টারনেট",          # internet (loanword)
    "মোবাইল",            # mobile (loanword)
    "প্রযুক্তি",          # technology
    "সরকারি",            # governmental
    "সরকারিভাবে",        # governmentally
    "ভালোবাসা",          # love
    "ভালোবাসার",         # of love
]


def probe_segmentation(tokenizer, words: list[str]) -> list[dict]:
    """Show how each tokenizer segments morphologically interesting words."""
    results = []
    for word in words:
        tokens = tokenizer.encode_str(word)
        ids = tokenizer.encode(word)
        results.append({
            "word": word,
            "tokens": tokens,
            "ids": ids,
            "n_tokens": len(tokens),
        })
    return results


def evaluate_one(tokenizer, texts: list[str]) -> dict:
    """Run all metrics on one tokenizer."""
    log.info("Evaluating: %s (vocab=%d)", tokenizer.name, tokenizer.vocab_size)

    fertility = compute_fertility(tokenizer, texts)
    compression = compute_compression_ratio(tokenizer, texts)
    avg_doc_tokens = compute_avg_doc_tokens(tokenizer, texts)
    script_dist = compute_script_distribution(tokenizer, texts)
    byte_rate = compute_byte_fallback_rate(tokenizer, texts)
    probes = probe_segmentation(tokenizer, PROBE_WORDS)

    # Avg tokens per probe word
    avg_probe_tokens = sum(p["n_tokens"] for p in probes) / len(probes)

    result = {
        "name": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "source": tokenizer.source,
        "fertility": round(fertility, 3),
        "compression_ratio": round(compression, 2),
        "avg_doc_tokens": round(avg_doc_tokens, 1),
        "byte_fallback_pct": round(byte_rate, 2),
        "script_distribution": script_dist,
        "avg_probe_tokens": round(avg_probe_tokens, 2),
        "probe_segmentation": probes,
    }

    log.info("  fertility=%.3f, compression=%.2f bytes/tok, byte_fb=%.2f%%",
             fertility, compression, byte_rate)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", required=True, help="Bengali JSONL for evaluation")
    parser.add_argument("--trained-dir", default=None, help="Dir with trained .model files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-docs", type=int, default=5000, help="Max docs to evaluate")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace tokenizers")
    args = parser.parse_args()

    # Load eval data
    log.info("Loading eval data from %s...", args.eval_data)
    texts = load_eval_texts(args.eval_data, args.max_docs)
    log.info("Loaded %d documents for evaluation.", len(texts))

    tokenizers = []

    # Load trained SentencePiece tokenizers
    if args.trained_dir:
        for model_file in sorted(Path(args.trained_dir).glob("*.model")):
            name = model_file.stem
            try:
                tokenizers.append(SPMTokenizer(str(model_file), name))
            except Exception as e:
                log.warning("Failed to load %s: %s", name, e)

    # Load existing Kotha-1 tokenizer
    kotha1_path = "/home/oneknight/projects/bangla-llm/tokenizer/output/bangla_bpe_32k.model"
    if os.path.exists(kotha1_path):
        tokenizers.append(SPMTokenizer(kotha1_path, "kotha1-bpe-32k"))

    # Load HuggingFace tokenizers for comparison
    if not args.skip_hf:
        hf_models = [
            ("meta-llama/Llama-3.2-1B", "llama-3.2"),
            ("Qwen/Qwen2.5-0.5B", "qwen-2.5"),
            ("google/gemma-3-1b-pt", "gemma-3"),
            ("bigscience/bloom-560m", "bloom"),
            ("csebuetnlp/banglabert", "banglabert"),
            ("csebuetnlp/banglat5", "banglat5"),
        ]

        for model_id, name in hf_models:
            try:
                log.info("Loading HF tokenizer: %s", model_id)
                tokenizers.append(HFTokenizer(model_id, name))
            except Exception as e:
                log.warning("Failed to load %s: %s", model_id, e)

    if not tokenizers:
        log.error("No tokenizers loaded. Aborting.")
        sys.exit(1)

    log.info("Evaluating %d tokenizers...", len(tokenizers))

    results = []
    for tok in tokenizers:
        try:
            result = evaluate_one(tok, texts)
            results.append(result)
        except Exception as e:
            log.error("Error evaluating %s: %s", tok.name, e)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("Results saved to %s", args.output)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Tokenizer':<20} {'Vocab':>7} {'Fertility':>10} {'Bytes/Tok':>10} "
          f"{'AvgDocTok':>10} {'ByteFB%':>8} {'AvgProbe':>9}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["fertility"]):
        print(f"{r['name']:<20} {r['vocab_size']:>7} {r['fertility']:>10.3f} "
              f"{r['compression_ratio']:>10.2f} {r['avg_doc_tokens']:>10.1f} "
              f"{r['byte_fallback_pct']:>8.2f} {r['avg_probe_tokens']:>9.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
