# Results notes

This directory contains frozen result artifacts that were useful during paper drafting.

## Included file

- `full_eval.json`: checked-in reference output from an earlier evaluator revision

## Important caveats

- The current evaluator CLI is cleaner and more portable than the revision that produced `full_eval.json`.
- The checked-in JSON should be treated as a transparency artifact, not as a strict schema contract.
- The paper table and the checked-in JSON are related but not identical artifacts; the paper comparison also included external baselines that were not all preserved as a single regenerated JSON file in this repository snapshot.

If you rerun the experiments, prefer the current scripts and record your own output file alongside the exact command line you used.
