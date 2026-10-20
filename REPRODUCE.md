# Reproducing the results

There are two levels, and they cost very different things.

**Level 1 — check the published numbers.** Minutes, on any machine, offline.
Everything needed is in this repository.

**Level 2 — rerun the experiment from generation.** Days of GPU time, a local
serving runtime, and the benchmark datasets. This regenerates the records that
level 1 checks. The generation records of our runs are not distributed; a rerun
writes its own under `data/`.

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

### The analyses added in the revision (release 1.1)

```
python3 scripts/verify_revision.py           # about 4 minutes on one core
python3 scripts/verify_revision.py --fast    # without the four re-fitting analyses
```

The revision added analyses in reply to the reviewers: output statistics over
all items (Figure 1), held-out problems, families and benchmarks, feature
groups, thresholds and vote rules, the token limit, per-item latency, four
backbones of further families, and stage-2 sampling at temperature 0.3. Each has
a script in `scripts/revision/` that writes one JSON into `results/revision/`;
`results/ssot.json` carries them in its `review_r1` block, and the manuscript
reads them from there. `verify_revision.py` runs every script into a temporary
directory and compares each recomputed number with `ssot.json`. It also
recomputes the MOME score of every further-family item from its 26 features
with the frozen predictor.

The scripts read these per-item files, written once from the generation records:

```
results/revision/items.jsonl                    the 4,497 evaluated items: c_safe,
                                                first-stage band, label, probe
                                                generations, and the ballot of the
                                                greedy answer plus k = 1..4 samples
results/revision/newfamily_items.jsonl          the 3,600 items of the four
                                                further-family backbones: the 26
                                                features, c_safe, label, mean
                                                log-probability, and the ballots at
                                                stage-2 temperature 0.7 and 0.3
results/revision/registered_temperature.jsonl   the five backbones of the main
                                                evaluation: the ballots of k = 2
                                                and k = 4 at 0.7 and at 0.3
results/revision/budget_b4096_items.jsonl       the 4096-token run of the MATH-500
                                                cells, k = 2
results/revision/latency_items.jsonl            seconds of every generation the
                                                gate buys, per item
results/revision/stage2_samples_items.jsonl     correctness of the greedy answer and
                                                of the first two samples at 0.7 and
                                                0.3, on the items sent to stage 2
model/frozen_predictor_newfamily.json           the predictor and tau_high that score
                                                the further-family backbones
```

A ballot is the vote share of the plurality answer and whether that answer is
correct, counted with the registered answer equivalence. The re-fitting
analyses read the features in `results/fingerprints.jsonl`; on them, the
backbone-out refit reproduces the registered scores. The frozen
further-family predictor was fitted once, before any further-family answer
was generated, on features rebuilt from the generation records at the time.
Those differed from `results/fingerprints.jsonl` on some MATH-500 items (the
symbolic comparison has a wall-clock timeout, see below), so the fit itself is
not repeated here; its coefficients, tau_high and the SHA-256 of its fitting
rows are in the model file.

## Level 2: rerun the experiment

### What you need

- A local serving runtime exposing generated tokens and the top token
  log-probabilities, with the five quantized backbones loaded.
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
