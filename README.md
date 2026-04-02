# Subword Tokenization Efficiency for Bengali Language Modeling

Code, data, and trained tokenizers for the paper:

**"Subword Tokenization Efficiency for Bengali Language Modeling"**
Shifat Islam Santo, The University of Texas at Dallas

[Paper (PDF)](paper/paper.pdf)

## Key Findings

- Bengali-dedicated tokenizers (1.47-1.75 tokens/word) are **5-9x more efficient** than most multilingual LLM tokenizers (7.32-13.69 tokens/word) on Bengali text
- A **single missing Unicode combining character** (U+09BC Bengali Nukta) accounts for 89.8% of byte-fallback tokens in affected tokenizers, inflating rates from 2% to 20%
- BLOOM (1.75) is a notable exception among multilingual models, achieving competitive Bengali efficiency due to its 251K vocabulary with Bengali coverage

![Fertility comparison](figures/fertility_comparison.png)

## Repository Structure

```
bengali-tokenizer-eval/
  paper/              # LaTeX source + compiled PDF
  scripts/            # Tokenizer training and evaluation code
  results/            # Raw evaluation results (JSON)
  trained_tokenizers/ # 13 SentencePiece .model files
  figures/            # Generated figures
```

## Trained Tokenizers

13 SentencePiece tokenizers in `trained_tokenizers/`:

| Name | Algorithm | Vocab | Data Mix |
|------|-----------|-------|----------|
| bn-bpe-16k | BPE | 16K | Bengali |
| bn-bpe-32k | BPE | 32K | Bengali |
| bn-bpe-49k | BPE | 49K | Bengali |
| bn-bpe-64k | BPE | 64K | Bengali |
| bn-uni-16k | Unigram | 16K | Bengali |
| bn-uni-32k | Unigram | 32K | Bengali |
| bn-uni-49k | Unigram | 49K | Bengali |
| bn-uni-64k | Unigram | 64K | Bengali |
| mix-bpe-32k | BPE | 32K | Bengali+English |
| mix-bpe-49k | BPE | 49K | Bengali+English |
| mix-bpe-64k | BPE | 64K | Bengali+English |
| mix-uni-32k | Unigram | 32K | Bengali+English |
| mix-uni-49k | Unigram | 49K | Bengali+English |

## Reproducing Results

### Train tokenizers

```bash
python scripts/train_all_tokenizers.py \
    --bn-corpus /path/to/bengali.jsonl \
    --en-corpus /path/to/english.jsonl \
    --output-dir trained_tokenizers \
    --sample-gb 1.5
```

### Evaluate tokenizers

```bash
python scripts/evaluate_tokenizers.py \
    --eval-data /path/to/eval.jsonl \
    --trained-dir trained_tokenizers \
    --output results/eval.json \
    --max-docs 3000
```

### Dependencies

```
sentencepiece
transformers
torch
```

## Evaluation Data

The 3,000-document Bengali evaluation set is sampled from a 1.2M-document web corpus (CulturaX + Bengali Wikipedia + Sangraha). Due to size (62MB), it is hosted separately on HuggingFace Datasets (link TBD).

## Citation

```bibtex
@article{santo2026bengali,
    title={Subword Tokenization Efficiency for Bengali Language Modeling},
    author={Santo, Shifat Islam},
    journal={arXiv preprint},
    year={2026}
}
```

## License

MIT
