# Reproducing the results

There are two levels, and they cost very different things.

**Level 1 — check the published numbers.** Minutes, on any machine, offline.
Everything needed is in this repository.

**Level 2 — rerun the experiment from generation.** Days of GPU time, a local
serving runtime, and the benchmark datasets. This regenerates the records that
level 1 checks.

---

## Level 1: check the published numbers

```
python3 scripts/verify_results.py
python3 scripts/figures/make_all.py
PYTHONPATH=mome python3 -m unittest discover -s tests
```

`verify_results.py` derives every published number again rather than reading
it back:

| what it checks | how |
|---|---|
| the frozen predictor | recomputes `c_safe` for all 4,497 items from `results/fingerprints.jsonl` and `model/s_v1_backbone_out.json`, and compares with the score recorded at decision time |
| the thresholds | puts each recomputed score through `model/thresholds.json` and compares the band with the recorded decision |
| coverage, selective accuracy, correct yield | recounts them from `results/per_item.jsonl` |
| the confusion matrix | recounts each row, and checks that rescued − lost − withheld equals the difference in correct yield |
| the AUROC panel | recomputes all five from `results/s0_rows.jsonl` |
| the intervals | reruns the stratified bootstrap, 2,000 replicates, seed 0 |

The files it reads are small on purpose:

```
results/ssot.json               the published numbers, all of them
results/per_item.jsonl          one row per evaluated item: cell, label,
                                c_safe, first-stage band, final state,
                                served, correct, truncated, mean logprob
results/fingerprints.jsonl      the 26 features per item
results/s0_rows.jsonl           the two served strata and their statistics
results/stage1_decisions.jsonl  the first-stage decision for every item
model/                          the frozen predictors and thresholds
```

## Level 2: rerun the experiment

### What you need

- A local serving runtime exposing generated tokens and the top token
  log-probabilities, with the five quantized backbones loaded.
- The generation records, published with the archived dataset. Unpack them
  under `data/` following the layout in `mome/paths.py`.
- The benchmark items. They are not redistributed here; rebuild them from the
  public sources into `data/benchmarks/`. The probe manifests in
  `data/manifests/` carry a SHA-256 of every edited problem, so a rebuild that
  differs from ours is detected rather than silently used.
- `pip install -r requirements.txt`

### The pipeline

```
# 1. Probes: rule-generated, no model call. Checks its own manifests.
python3 mome/build_probe_sets.py

# 2. Generation. One greedy answer per item, then one per probe.
python3 mome/run_campaign.py --config configs/<cell>.json

# 3. Fingerprints: join originals with probe responses, read the features.
python3 mome/fingerprint.py --all

# 4. The feature matrix, and the predictor fitted on it.
python3 mome/build_dataset.py
python3 mome/evaluate_s.py
python3 mome/thresholds.py

# 5. Apply the first stage; write the second stage's work list.
python3 mome/select_retry.py

# 6. The second stage's samples, then grade them.
python3 mome/run_pool.py --config configs/retry/<cell>.json
python3 mome/grade_retry.py

# 7. The registered readout, run once.
python3 mome/evaluate_gate.py

# 8. Regenerate every published number, and verify them.
python3 scripts/export_ssot.py
```

Step 8 ends by calling `verify_results.py`, so a regenerated file that does not
reproduce is caught where it is written rather than after it reaches a paper.

### What is fixed before the data is seen

These are not conventions this code happens to follow; they were registered
before the corresponding generation was run, and the code was written to make
departing from them visible.

- **The predictor never sees its own backbone.** Evaluation scores each
  backbone with coefficients fitted on the other four. The shipped predictor
  (`s_v1.json`) is fitted on everything and carries a calibration layer; it is
  *not* what the reported numbers were computed with. Mixing the two shifts the
  bands badly, and `select_retry.py` says so at the point where it matters.
- **Thresholds are chosen on the training side only**, by an inner
  leave-one-backbone-out CV, and the held-out backbone is read exactly once.
- **The probe set is pinned by hash.** Regenerating probes that differ from the
  committed manifest stops the run. Changing the probe set is a deliberate act:
  delete the manifest, regenerate, commit.
- **The adoption rule is stated in advance**: among variants whose correct
  yield is at least the ungated model's, take the highest selective accuracy;
  on a tie, the cheaper one. If none clears the bar, the result is reported as
  a failure of capability preservation, and the rule is not changed afterwards.
- **Nothing is graded before the gate decides.** The gate reads no correct
  answer; grading happens strictly afterwards, with the same grader every
  campaign used.

### Two things that will not reproduce exactly

- **Symbolic comparison carries a wall-clock timeout.** A slower machine can
  return a different verdict on a hard pair. The registered thresholds
  reproduced exactly in our re-runs, but a per-item fingerprint can differ in
  the last digit across machines. A step-counted budget would be
  machine-independent and is the better design; the comparator exposes no step
  counter.
- **Generation is greedy but not bit-identical across runtime versions.** The
  first stage uses temperature 0, so it carries no seed dependence; the second
  stage samples at 0.7 with seeds recorded in its configs.

## Layout

`mome/paths.py` defines every directory this package reads or writes. It is the
one place to look, and the one place to change if you lay the data out
differently.
