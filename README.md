# Missed Delivery Early Warning System

Master's Thesis project that investigates an early-warning system for delivery
obligations at risk of a `missed_delivery`, with an intended prediction horizon
of seven calendar days before their target date (D-7). The current results are
reproducible retrospective evidence; strict D-7 validity remains unverified
because of the temporal input limitations described below.

The project combines a reproducible scientific workflow, a frozen supervised
learning model, a reusable local inference package, and a Microsoft Fabric
batch-scoring prototype. Its purpose is decision support: prioritising cases
for operational review. It is not an autonomous delivery-management system
and does not replace business judgement.

## Project navigation

- `notebooks/scientific/`: complete analytical workflow from data understanding to final evaluation;
- `src/missed_delivery/`: reusable feature engineering, validation, backtesting, and inference code;
- `notebooks/fabric/` and `fabric/pipelines/`: sanitised Fabric batch prototype;
- [Final submission report](docs/revision_tfm/overleaf_ampliado/final_report.tex): expanded Master's Thesis report matching the final Overleaf ZIP;
- `models/` and `configs/`: machine-readable feature, model, and threshold contracts;
- `data/`: reproducible SQL extraction and scoring contracts; private CSV files remain Git-ignored.

## Research problem and objective

Missed deliveries can disrupt production planning, increase logistics costs,
and affect customer service. Historical reporting explains failures only after
they occur; this project investigates whether information available before the
delivery date can identify elevated risk early enough to support intervention.

The main objective is to estimate, at D-7, the probability that a planning
group will register any lost quantity on its target date. The work addresses
four questions:

1. Can missed deliveries be predicted from pre-event operational history?
2. Which model provides the best balance between discrimination, calibration,
   and operational usefulness?
3. Can a decision threshold achieve the required Recall without exceeding the
   expected daily review capacity?
4. Can the frozen local model be reproduced faithfully in Microsoft Fabric?

## Analytical definition

The analytical unit is one observation per:

```text
target_date + Grupo_raiz
```

`Grupo_raiz` is the planning-group identifier used to aggregate the underlying
delivery obligations. The binary target is:

```text
missed_delivery = 1 if lost > 0, otherwise 0
```

This definition captures the occurrence of any missed quantity. It does not
encode financial severity or impose a minimum materiality threshold. Requested,
lost, delivered, and other post-event outcome fields are excluded from the
predictors.

## Dataset

The analytical dataset contains:

- 33,971 observations between 2025-01-02 and 2026-08-25;
- 2,715 positive cases;
- an observed event rate of 7.99%;
- production, demand, stock, fulfilment, planning, and contextual information.

Company data and fitted binary artifacts are intentionally excluded from the
public repository. The SQL contracts, schemas, metadata, notebooks, and source
code are retained so that the workflow can be audited and reproduced in an
authorised environment. Stored notebook outputs are cleared from the public
delivery to avoid publishing operational identifiers; the notebooks can be
rerun where the authorised data and Fabric resources are available.

## Feature engineering

The frozen model expects exactly 154 features in the order stored in
`models/feature_schema.json`:

| Feature family | Count | Description |
|---|---:|---|
| Context | 4 | `Grupo_raiz`, customer, UAT, and destination |
| Historical availability | 6 | Exact-lag availability and group-history support |
| Operational lags | 72 | 18 operational variables at D-7, D-14, D-21, and D-28 |
| Historical gaps | 24 | Production, demand, stock, and accumulated-plan differences |
| Temporal deltas | 45 | Short-, medium-, and wider-horizon changes |
| Calendar | 3 | Month, Monday-based day of week, and ISO week |
| **Total** | **154** | Frozen model input contract |

All lag joins require equality on `Grupo_raiz` and the exact calendar date.
A D-13 record cannot replace a missing D-14 record. Missing exact history is
preserved as missing, with explicit availability features; no nearest-date,
forward-fill, or interpolation rule is used.

`history_records_before_D` counts distinct historical dates for the same group
strictly before the target date. This includes dates after the D-7 scoring
cutoff and is a temporal availability defect relative to the intended horizon.
The comparison documented in the final report changes this feature in 33,917
of 33,971 observations (99.84%), including 5,647 holdout rows, when history is
restricted to dates at or before D-7. Categorical context is also extracted
from realised historical delivery records and requires point-in-time
verification. Exact lag matching and excluding outcome columns do not by
themselves establish availability of every predictor at scoring time.

Derived gaps and deltas preserve missingness when an operand is unavailable.

## Scientific design

The project uses chronological separation rather than a random split:

| Stage | Period | Purpose |
|---|---|---|
| Training | 2025-01-02 to 2026-01-31 | Fit preprocessing and candidate models |
| Calibration | 2026-02-02 to 2026-03-31 | Fit the frozen Sigmoid calibrator |
| Calibration-method evaluation and threshold selection | 2026-04-01 to 2026-04-30 | Evaluate calibration methods and select an operational decision rule |
| Holdout | 2026-05-04 to 2026-08-25 | Final comparative evaluation |

Model development includes temporal validation and expanding-window
backtesting. The selected configuration is **XGBoost Full**. Missing values are
handled within the fitted pipeline, while categorical preprocessing is fixed
inside the serialized model.

The holdout requires an important qualification: it was inspected during the
initial Logistic Regression baseline work. It is therefore useful temporal
evaluation evidence but is not claimed as a strictly untouched test set. The
prospective evaluation must enforce point-in-time inputs and retain matched
predictions and mature outcomes before it can provide verified blind evidence.

The historical-support cutoff must be corrected and categorical availability
verified before claiming strict D-7 performance. Changing the feature definition
requires coherent reevaluation of training, calibration and inference, and may
require refitting and recalibration. Its impact on the reported metrics has
not been measured; the frozen artifacts currently preserve the evaluated
configuration.

## Calibration and decision threshold

The model's inference contract is:

```text
154 ordered features
-> frozen XGBoost pipeline
-> raw_score
-> clip to [1e-6, 1-1e-6]
-> logit(raw_score)
-> frozen Sigmoid calibrator
-> calibrated_probability
-> threshold 0.08
-> alert_flag
```

The 0.08 threshold was selected only on the April threshold period under a
minimum-Recall and review-capacity criterion. It remains
`provisional_pending_business_approval`: Operations must validate the Recall
objective, daily workload, and relative false-negative/false-positive costs
before production use.

## Holdout results

| Metric | Result |
|---|---:|
| Precision | 0.2328 |
| Recall | 0.7417 |
| F1 | 0.3543 |
| PR-AUC | 0.2869 |
| ROC-AUC | 0.8516 |
| Brier Score | 0.0514 |
| Alert rate | 0.2013 |
| Alerts per active day | 16.62 |
| True positives | 267 |
| False positives | 880 |
| False negatives | 93 |
| True negatives | 4,457 |

At threshold 0.08 the model detects 267 of 360 positive holdout observations,
with 23.28% Precision and 74.17% Recall. The average of 16.62 alerts per active
day is above the provisional planning constraint of 15. This measured trade-off
is why the threshold is technically selected but not yet business-approved.

PR-AUC is computed with `average_precision_score`, rather than trapezoidal
integration. These results are reproducible for the current feature matrix;
they are not a fully validated estimate of performance with inputs available
at D-7.

## Local implementation and reproducibility

The reusable Python package separates feature construction, validation, and
inference from notebook execution:

```python
from missed_delivery.features import build_temporal_features
from missed_delivery.inference import predict_batch

features = build_temporal_features(scoring_universe, operational_history)
predictions = predict_batch(features)
```

The local implementation validates the analytical key, rejects duplicate
history keys, builds exact lags, preserves missing values, selects columns in
the frozen order, and applies the calibrated scoring chain.

The supported repository environment is Python 3.11 or newer and is validated
with Python 3.12. Install and run the checks with:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The current suite contains 21 tests covering schema validation, forbidden-column checks,
exact-lag semantics, missing history, unseen categories, frozen inference, and
the Fabric registration wrapper. All 21 tests pass in the final reviewed state.
These checks do not establish complete point-in-time feature availability or
end-to-end local/Fabric feature parity. In particular, the historical-support
cutoff defect remains despite the passing suite.

Complete inference tests require the private files under `models/` and
`data/processed/` to be present in the authorised local environment.

## Microsoft Fabric prototype

The Fabric notebooks cover model registration, feature generation, and batch
scoring:

- Notebook 07 registers the frozen model through MLflow;
- Notebook 08 reconstructs post-holdout order snapshots and prepares scoring features in Spark;
- Notebook 09 loads the registered model and writes prediction results.

The current registered model is `md_early_warning`, **Version 2**. Registry
Version 1 is retained only for traceability. Version 2 corrects the calibration
wrapper so that it applies the required clip and logit transformation before
the Sigmoid calibrator. Notebook 07 contains only this corrected wrapper and
loads the logged model explicitly before its parity check.

The documented Fabric batch execution recorded:

- 101 scoring observations;
- 154 ordered model features;
- four outputs: raw score, calibrated probability, threshold, and alert flag;
- 14 alerts and 87 non-alerts;
- model name, Version 2, run ID, and UTC run timestamp.

This run demonstrates batch execution and output structure. It does not
establish predictive precision or verified blind validation. The post-holdout
snapshot assessment still requires retained matched predictions, mature labels,
explicit handling of unmatched outcomes and a reproducible evaluation procedure.

The registered Version 2 output matches canonical local inference for the
stored registration example. Nevertheless, full local/Fabric feature parity
is not yet demonstrated. Notebook 08 currently derives `customer`, `uat`, and
`destination` from historical `Fact_OTD` records rather than from the active
obligations available at the scoring cutoff. Its historical-support count also
uses dates before the target date, without enforcing the scoring cutoff. A golden test comparing all 154
features and four predictions for identical observations is also still
required.

The sanitised Fabric pipeline template orchestrates the implemented batch flow:

```text
scoring_features
    -> Refresh SQL analytics endpoint
    -> Wait
    -> model_prediction
    -> Teams notification
```

The template contains placeholders instead of tenant-specific workspace,
notebook, endpoint, chat, and object identifiers. It must be rebound after
import and does not include a production schedule or monitoring layer.

Accordingly, Fabric is a functioning **batch-scoring prototype with
conditions**, not a fully governed production deployment. Production
ownership, monitoring, and operational approval remain outside this delivery.

## Daily scoring population

The future scoring universe must come from a governed source of active delivery
obligations. It must not be built from `Fact_OTD` or another table containing
the realised `lost/delivered` outcome.

Eligible rows satisfy the seven-calendar-day horizon:

```text
target_date = scoring_date + 7 calendar days
```

Multiple physical obligations are aggregated to the frozen analytical unit.
Notebook 08 implements a historical snapshot path using
`FACT_ENVIOS_PREV_HIST`; this demonstrates population reconstruction, while
production mappings for open status, cancellation, remaining quantity, daily
cutoffs and late updates still require verification. Its inner context join
can also exclude obligations without matching context.
The contract and conceptual SQL view are documented in:

- `docs/daily_scoring_universe.md`;
- `data/md_daily_scoring_universe_query.sql`.

## Scope and limitations

The repository supports the following claims:

- the modelling workflow uses chronological evaluation and exact operational lags;
- the frozen local inference path is reproducible when private artifacts are
  available;
- Fabric Version 2 reproduces the canonical calibration transformation;
- a real 101-row Fabric scoring run completed with the expected schema.

It does not yet support these stronger claims:

- that all predictors were available at D-7 or that the reported metrics establish strict D-7 performance;
- that the holdout is strictly blind;
- that the post-holdout snapshot workflow has provided verified blind predictive evidence;
- that threshold 0.08 has final operational approval;
- that all 154 Fabric features have been proven equal to local values row by
  row;
- that the solution is automated, monitored, or production-ready.

These limitations include a scientific input-availability defect as well as
engineering and governance conditions. Correcting temporal inputs requires
coherent model reevaluation and may require refitting and recalibration. The
binary target remains the occurrence of any positive lost quantity.

## Repository structure

```text
configs/                 threshold configuration and governance
data/                    SQL contracts and Git-ignored local datasets
models/                  frozen artifacts, feature schema, and metadata
notebooks/scientific/    scientific workflow 01–06
notebooks/fabric/        Microsoft Fabric prototype 07–09
fabric/pipelines/         sanitised Fabric pipeline template
src/missed_delivery/     reusable feature, validation, and inference code
tests/                   schema, leakage, temporal, and parity tests
docs/                    scoring-universe contract, final report and presentation materials
report/figures/           analytical figures referenced by the presentation guide
```

The final submission source is
[docs/revision_tfm/overleaf_ampliado/final_report.tex](docs/revision_tfm/overleaf_ampliado/final_report.tex),
matching `TFM_Overleaf_ampliado_Fabric_AI_Engineering.zip`. Superseded reports,
working review notes and slide previews have been removed from the delivery.
The final source contains 30 cited references. The
[compiled submission PDF](docs/revision_tfm/overleaf_ampliado/final_report.pdf)
has 15 pages including the title page, abstract, contents and references,
verified on 13 September 2026 with Tectonic (XeTeX). Main sections start on
new pages; related topics are grouped as subsections without removing the
scientific text. The body uses the included Times New Roman font at 11 points.
The Overleaf ZIP includes the compiled PDF and sources. Select XeLaTeX and
`final_report.tex` in Overleaf, and confirm the page count after recompilation
with the selected TeX Live version.

## Final status

The project provides a reproducible retrospective modelling workflow and a
Fabric batch-scoring proof of concept. Strict D-7 validity requires correcting
the historical-support cutoff, verifying point-in-time categorical context and
reevaluating the pipeline. Full local/Fabric equivalence additionally requires
a retained golden comparison of all 154 features and four prediction outputs.
Prospective evaluation, business approval of the threshold, monitoring and
production ownership remain pending.
