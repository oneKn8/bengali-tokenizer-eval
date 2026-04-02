# Launch kit

Copy-paste assets for announcing the repository, paper, and release.

## GitHub repo description

Tokenizer choice can make Bengali 5-9x more expensive. Paper, evaluation framework, Unicode failure analysis, and 13 released tokenizers.

## GitHub release title

Research release: Bengali tokenizer evaluation + 13 SentencePiece models

## GitHub release body

This release packages a systematic evaluation of tokenizer efficiency for Bengali language modeling.

### Why it matters

- Bengali tokenizer choice changes efficiency dramatically: dedicated tokenizers achieve 1.47-1.75 tokens/word, while most multilingual LLM tokenizers need 7.32-13.69.
- In practice, that means a fixed context window can cover 5-9x less Bengali text depending on the tokenizer.
- A single Unicode combining character, U+09BC Bengali Nukta, explains most byte-fallback inflation in the affected training runs.

### Included in the repo

- paper source and PDF
- evaluation scripts
- 13 released SentencePiece tokenizers
- frozen result artifacts
- portability improvements for reproducing paper-style comparisons

### Who this helps

- Bengali LLM builders
- multilingual model practitioners evaluating tokenizer bottlenecks
- researchers working on Indic NLP, tokenization, and Unicode robustness

Repo: https://github.com/oneKn8/bengali-tokenizer-eval

## X thread

1. I released a benchmark repo for Bengali tokenizer efficiency. The main result: tokenizer choice can make Bengali 5-9x more expensive in context usage.

2. On 3,000 Bengali documents, dedicated Bengali tokenizers land at 1.47-1.75 tokens/word. Most multilingual LLM tokenizers land at 7.32-13.69.

3. Same model budget, radically different usable context. A 4,096-token window can hold roughly 2,800 Bengali words with a strong tokenizer, or only about 300 with a weak one.

4. I also found a sharp Unicode failure mode in my own training runs: one missing combining character, U+09BC Bengali Nukta, explained 89.8% of byte-fallback tokens.

5. That one character pushed byte fallback from about 2% to about 20%. Small Unicode mistakes can wreck tokenizer efficiency at scale.

6. The repo includes the paper, evaluation code, 13 released SentencePiece tokenizers, figures, and a portable comparison workflow.

7. If you're building Bengali LLMs, doing continued pretraining, or evaluating multilingual tokenizers on Indic text, this is the baseline I wish existed earlier.

8. Repo: https://github.com/oneKn8/bengali-tokenizer-eval

## LinkedIn post

I just released a benchmark repo on Bengali tokenizer efficiency, and the result is bigger than I expected:

Tokenizer choice can make Bengali 5-9x more expensive in effective context usage.

Across 3,000 Bengali documents, Bengali-dedicated tokenizers achieved 1.47-1.75 tokens per word. Most multilingual LLM tokenizers needed 7.32-13.69.

That means a fixed context window can cover dramatically less Bengali text depending on the tokenizer you inherit.

One of the most interesting findings was a Unicode failure mode in my own training runs: a single missing combining character, U+09BC Bengali Nukta, explained 89.8% of byte-fallback tokens and pushed byte fallback from about 2% to about 20%.

I released:

- the paper
- the evaluation framework
- 13 SentencePiece tokenizers
- result artifacts and reproduction notes

If you work on Bengali NLP, multilingual LLMs, tokenizer design, or Unicode-heavy scripts, I think this repo will be useful.

Repo: https://github.com/oneKn8/bengali-tokenizer-eval

## Hacker News title options

- Show HN: Bengali tokenizer choice can change context efficiency by 5-9x
- I released a Bengali tokenizer evaluation repo with 13 models and a Unicode failure analysis
- Tokenizer choice can make Bengali 5-9x more expensive

## Short launch blurb

Released: a benchmark repo for Bengali tokenizer efficiency. It includes the paper, evaluation framework, 13 tokenizers, and a Unicode failure analysis showing how one missing combining character can blow up byte fallback.
