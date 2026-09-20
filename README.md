# MOME — Metamorphic Evidence as a Runtime Reliability Signal

Reference implementation and reproduction package for *Metamorphic Evidence as
a Runtime Reliability Signal for Small Language Models*.

A quantized small language model is served behind a runtime that returns
generated tokens and token log-probabilities and nothing else. There is no
hidden state to read, no larger model to escalate to, and no label at
inference. **Which observable signal says the answer is wrong?**

The statistics such a runtime does expose answer it badly. Over 4,497 evaluated
items, the best of five output-derived confidence proxies separates served
correct answers from served wrong ones at an AUROC of 0.607 — better than
chance, and far too weak to select on.

MOME reads a behavioural signal instead. It perturbs each problem in two
directions by rule — a restatement whose answer **must hold**, and an edit
whose answer **must change** — and observes whether the answers move with the
meaning. The responses form a metamorphic fingerprint, a fixed predictor turns
that fingerprint into a score, and a two-stage policy turns the score into
serve, verify or decline.

No model is opened, no second network is loaded, and no correct answer is read
at inference time.

## Results

Five quantized backbones (1.5–11 B) × three mathematical benchmarks in English
and Korean, 15 cells of 300 items each.

| | coverage | selective accuracy | correct yield | generations/item |
|---|---:|---:|---:|---:|
| ungated model | 100.0% | 69.5% | 69.5% | 1.00 |
| log-probability gate, matched coverage | 81.2% | 79.0% | 64.2% | 1.00 |
| **MOME** | **81.2%** | **85.8%** | **69.6%** | **4.34** |

Selective accuracy rises by 16.3 points while correct yield is preserved. The
preservation is not a wash of large opposing effects: across 4,497 items the
second stage rescued 116 wrong answers, lost 23 right ones, and declining
withheld 88 that would have been correct — a net of +5 items, which is the
+0.1 point of the table read at single-item resolution.

## Check the results yourself

No GPU, no model, no network, no large download:

```
python3 scripts/verify_results.py
```

This recomputes rather than re-reads. It re-derives `c_safe` for every item
from the 26 fingerprint features and the frozen coefficients, puts the scores
through the registered thresholds, recounts every rate from the per-item
outcomes, reruns the stratified bootstrap, and checks all of it against
`results/ssot.json`. 53 checks; any mismatch is a failure.

Redraw every figure in the paper from the published results:

```
python3 scripts/figures/make_all.py
```

Run the unit tests for the rule engine, the fingerprint assembler and the
gate's decision logic:

```
PYTHONPATH=mome python3 -m unittest discover -s tests
```

Reproducing the experiment from generation onward needs the generation records
and a local serving runtime; see [REPRODUCE.md](REPRODUCE.md). The records are
archived with this repository at
[doi:10.5281/zenodo.22854894](https://doi.org/10.5281/zenodo.22854894).

## What is here

```
mome/       the method
  perturb.py          the rule engine: MUST_HOLD and MUST_CHANGE probes
  fingerprint.py      the 26 observation features, assembled per item
  gate.py             c_safe, and the first stage's three bands
  gate2_features.py   what the second stage's samples expose
  build_dataset.py    the feature matrix the predictor is fitted on
  thresholds.py       threshold selection, on the training side only
  select_retry.py     apply the first stage; write the second stage's work
  evaluate_gate.py    the registered readout, run once
  run_campaign.py     generation: one greedy answer per item
  run_pool.py         generation: the second stage's sampled paths
  grade.py            the grader every campaign used
model/      the frozen predictors and the registered thresholds
results/    the published outputs, and the per-item records behind them
scripts/    verification, the figures, and the tools that regenerate results
tests/      unit tests, no data files needed
data/       probe manifests; the generation records go here (see REPRODUCE.md)
```

`results/ssot.json` is the single source of truth. Every table and every number
in the paper is generated from it, so no figure in the manuscript was ever
typed by hand.

## Requirements

Verification and the figures need Python 3.11+, and `matplotlib` for the
figures. Refitting the predictor additionally needs `numpy` and
`scikit-learn`; regenerating the results needs `math-verify` for symbolic
answer comparison. See `requirements.txt`.

The deployed path — the rule engine, the fingerprint and the score — uses the
standard library only.

## Citation

Archived at [doi:10.5281/zenodo.22854894](https://doi.org/10.5281/zenodo.22854894),
which resolves to the current version. See [CITATION.cff](CITATION.cff).

## License

Apache License 2.0 for the code; see [LICENSE](LICENSE). The benchmark
datasets are the property of their respective authors and are not
redistributed here.
