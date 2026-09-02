# Dissertation artifact release

This repository's `dissertation-v1.0` tag and its GitHub Release form the
reproducibility snapshot for the dissertation evaluation. The source code,
configuration, compact summaries and selected figures are in the tagged Git
tree. Large generated artifacts are provided in the Release asset
`smartgrid-fdi-edge-artifacts-dissertation-v1.0.zip`.

Release URL:

<https://github.com/pengbo960/Smartgrid-fdi-edge/releases/tag/dissertation-v1.0>

## Bundle contents

- `data/raw/training_runs/`: 30 MQTT collection CSV files (five independent
  runs for each of normal, constant, random, replay, topic spoof and gradual).
- `data/processed/multiview_dataset.csv`: the 61-column feature dataset used by
  the formal experiments.
- `data/samples/normal_sample.csv`: a small normal-data example for inspecting
  the raw MQTT schema without opening a full training run.
- `models/`: the fixed Joblib deployment artifacts and ordered feature/model
  metadata. Joblib files use Python pickle semantics and should only be loaded
  from this trusted release.
- `results/`: machine-readable CSV/JSON/TXT outputs, including predictions,
  grouped-fold summaries, deployment parity records, live MQTT logs and all
  repeated MacBook/Raspberry Pi drift logs. PNG figures are omitted because
  they are already present in the tagged source tree or can be regenerated.
- `MANIFEST.csv`: relative path, byte size, SHA-256 digest and category for
  every payload file.
- `SHA256SUMS`: SHA-256 digests for every file in the extracted bundle except
  `SHA256SUMS` itself.
- `SOURCE_COMMIT.txt`: the exact tagged source commit.

The repository's local `tmp/` directory, operating-system metadata and the
dissertation working PDF are deliberately excluded.

## Dataset audit

The intended schedule was 30 runs × 600 cycles × 3 devices = 54,000 messages.
The recorded dataset contains 53,388 messages: 17,796 complete three-device
cycles, or 98.87% of the nominal schedule. The 612-message difference is 204
unexecuted terminal cycles × 3 devices. Scenario execution is bounded by
elapsed wall-clock duration rather than an exact 600-iteration counter, so
printing, scheduling and MQTT overhead resulted in 591–596 completed cycles per
run. Device counts remain exactly balanced at 17,796 each, and no file ends
with a partial three-device cycle. All configured attack intervals are fully
present; the shortfall is confined to post-attack normal tails.

Raw and processed counts match exactly. Both contain 53,388 rows: 45,888 normal
rows and 7,500 attack rows. Each of the five attack families contributes 1,500
attack rows. All 118 published CSV files passed a full row/column consistency
parse before packaging, and all included JSON files parsed successfully.

## Meaning of unseen and first unknown

In the open-set evaluation, `unseen attack` means a withheld attack family with
respect to every fitted component. Constant, random, replay and topic spoof are
known classes; all gradual source files are excluded before preprocessing and
fitting, including their normal contexts and unaffected-device rows. Fold `i`
then tests the 300 labelled gradual attack messages from gradual run `i`. Thus
`unseen` means unseen by the trained classifier, scaler and anomaly detector;
it does not mean that the attack family was unknown to the experimenter.

The reported first-unknown values of 0, 3, 5, 1 and 0 attack steps have a mean
of 1.80 and a sample standard deviation of 2.17 steps. This metric is the first
single `unknown` decision in each fold, not a sustained-detection delay. In
addition, gradual step 0 has zero injected voltage bias. The value is therefore
best read as an early-response indicator under this synthetic trace, not as a
robust operational detection-time guarantee.

## Live MQTT evaluation denominators

The formal Raspberry Pi live evaluation contains 3,228 messages across five
scenarios. Constant, replay and topic spoof contribute 90 known-attack messages
and 990 matched normal messages; random was not included in this formal live
set. The known-attack alert rate is 90/90 (100%), and exact known-class
classification is 87/90 (96.67%). The extended gradual run contributes 300
attack messages and 1,488 normal messages; 280/300 attacks are labelled
`unknown` (93.33%). Across all five scenarios, the pooled normal alert rate is
51/2,838 (1.80%). Each simulator uses three devices at a 0.5-second per-device
publishing interval. The live detector parameters are fixed by
`config/raspberry_pi_edge.yaml` and the tagged model metadata.

## Live drift evaluation denominators and parameters

Each of five Raspberry Pi measurement-drift runs applies +5 V to `meter_02`
for steps 150–449: 300 active messages per run. The raw/guarded active-alert
counts are 62/47, 48/42, 55/44, 58/48 and 46/36, giving a mean per-run reduction
of 19.13% ± 4.49% (sample standard deviation). Each of five communication-drift
runs changes all three devices from 0.5 s to 0.8 s for steps 150–599: 1,350
active device messages per run. Every run records 1,347 raw alerts and 12
guarded alerts, a 99.11% reduction, with a five-message-per-device detection
delay.

The controlled Raspberry Pi drift configuration uses a 30-message candidate
window, minimum 20 trusted samples, blend factor 0.5 and maximum reference
update of 1.0 per approval. Voltage monitoring uses delta 0.05, threshold 220
and minimum 50 observations; source-interval monitoring uses delta 0.01,
threshold 2 and minimum 50 observations. Experimental auto-approval is enabled,
expires after 500 device-feature messages, requires confidence at least 0.98,
anomaly score at most 0.75 and history count at least 20, and permits a bounded
override. These settings are in
`config/raspberry_pi_edge_drift_experiment.yaml`; the production-style
`config/edge.yaml` keeps automatic approval disabled.

## Offline poisoning scope

The 1.21 V guarded versus 7.22 V unguarded reference-shift result is from a
separate simplified offline poisoning test. It uses 250 baseline samples at
230 V followed by 400 samples linearly increasing to 238 V, Gaussian noise with
standard deviation 0.25 V, a ±2 V trust gate around the fixed initial
reference, and Page-Hinkley parameters delta 0.02, threshold 8 and minimum 50
observations. It is not an end-to-end poisoning evaluation of the complete
Logistic Regression, Isolation Forest and MQTT protocol pipeline.

## Verification

After downloading the ZIP and its companion `.sha256` file, verify the archive
on macOS or Linux:

```bash
shasum -a 256 -c smartgrid-fdi-edge-artifacts-dissertation-v1.0.zip.sha256
unzip smartgrid-fdi-edge-artifacts-dissertation-v1.0.zip
cd smartgrid-fdi-edge-artifacts-dissertation-v1.0
shasum -a 256 -c SHA256SUMS
```

The bundle was produced from the tagged source using the package versions in
`requirements.txt` (Python 3.11, pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.9.0
and Joblib 1.5.3 among the direct dependencies). There is no DOI for this
release; the tagged GitHub Release URL above is the stable public locator.
