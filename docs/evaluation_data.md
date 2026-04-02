# Evaluation data notes

The released repository does not bundle the 3,000-document Bengali evaluation set directly.

## Expected input format

Both `scripts/train_all_tokenizers.py` and `scripts/evaluate_tokenizers.py` expect JSONL files where each line contains a `text` field:

```json
{"text": "বাংলা example document text ..."}
```

Extra fields are ignored by the scripts.

## Evaluation filter used by the current release

- Maximum documents: controlled by `--max-docs`
- Minimum document length: controlled by `--min-chars`
- Default release setting: `--min-chars 200`

## Corpus provenance

The paper evaluation set is described as a held-out Bengali web-text sample assembled from:

- CulturaX
- Bengali Wikipedia
- Sangraha

The paper also assumes standard Bengali cleaning and Unicode normalization before tokenizer training and evaluation.

## Why the JSONL is not bundled here

This repo release is intended to be publishable and transparent without pretending the benchmark file is already hosted in a finalized external location. Until the external hosting package is finalized, this document is the source of truth for the expected format and reconstruction recipe.

## Practical guidance

- If you reconstruct the benchmark yourself, keep the train and evaluation splits disjoint.
- If you compare against the paper numbers, keep `--max-docs 3000` and `--min-chars 200`.
- If you publish a hosted benchmark later, update the main README to link to that canonical location.
