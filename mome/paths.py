#!/usr/bin/env python3
"""Every directory this package reads or writes, defined once.

The modules here were written against a working tree whose layout carried
project-internal names.  Rather than leave those names scattered through
twenty files, each module now takes its directories from this one, so the
layout is stated in a single place and can be read at a glance:

    <root>/
      mome/          the code
      model/         frozen predictors and the registered thresholds
      configs/       generated campaign configurations
      data/          generation records -- large, fetched separately
        benchmarks/    benchmark item files, rebuilt locally
        items/         the item lists each cell ran
        manifests/     probe manifests (committed; they pin the probe set)
        ungated/       one directory per ungated cell
        probes/        one directory per probe cell
        fingerprints/  assembled fingerprints
        retry/         one directory per second-stage cell
      results/       published outputs and regenerated reports

Only `results/`, `model/` and the probe manifests are small enough to be
committed.  `data/` holds the generation records and is fetched separately;
see REPRODUCE.md.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = ROOT / "mome"
MODEL = ROOT / "model"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

#: Frozen predictors and thresholds.
MODEL_DEPLOYED = MODEL / "s_v1.json"
MODEL_BACKBONE_OUT = MODEL / "s_v1_backbone_out.json"
MODEL_STAGE2 = MODEL / "gate2_v2_symbolic.json"
THRESHOLDS = MODEL / "thresholds.json"

#: Generation records and their inputs.
BENCHMARKS = DATA / "benchmarks"
ITEMS = DATA / "items"
MANIFESTS = DATA / "manifests"
UNGATED = DATA / "ungated"
PROBES = DATA / "probes"
FINGERPRINTS = DATA / "fingerprints"
RETRY = DATA / "retry"

#: Generated campaign configurations.
PROBE_CONFIGS = CONFIGS / "probes"
RETRY_CONFIGS = CONFIGS / "retry"
RETRY_ITEMS = CONFIGS / "retry_items"

#: Published and regenerated outputs.
SSOT = RESULTS / "ssot.json"
OVERHEAD = RESULTS / "overhead.json"
SCORE_CARD = RESULTS / "score_model_card.md"
DECISIONS = RESULTS / "stage1_decisions.jsonl"
FEATURES = RESULTS / "fingerprints.jsonl"
PER_ITEM = RESULTS / "per_item.jsonl"
STRATA_ROWS = RESULTS / "s0_rows.jsonl"
LAYERS = RESULTS / "layers"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "figures"
