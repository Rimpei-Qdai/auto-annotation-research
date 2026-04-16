---
description: Sync the latest annotation run from kiwi, compare against manual labels, and write a dated report with metrics, causes, and next fixes.
---

# sync-annotation-report

Sync the latest auto-annotation artifacts from the remote server, compare them
against the manual labels, and write a dated evaluation report.

## Arguments

- `host`: remote host, default `kiwi`
- `manual_csv`: optional override for the manual label CSV
- `macro4_csv`: optional override for the 4-class helper CSV
- `report_suffix`: optional short suffix to append to the report filename

## Preflight

1. Confirm the current repo is `/Users/rimpeihata/Desktop/auto-annotation-research`.
2. Confirm remote artifacts exist on
   `<host>:~/workspace/auto-annotation-research/new/annotation_tool`:
   - `annotated_samples_auto.csv`
   - `annotation.log`
   - `inference_process.jsonl`
3. Default to `kiwi` if the user did not provide `host`.
4. Read the manual label files from:
   - `new/results/annotated_samples_manual.0121.csv`
   - `new/results/annotated_samples_manual.0121_macro4.csv` when present
5. Note any uncommitted local changes that could make the report ambiguous, but
   do not block on them.

## Plan

1. Sync the latest run artifacts from the remote host into
   `new/annotation_tool/kiwi_logs/`.
2. Refresh the top-level local `annotated_samples_auto.csv`.
3. Evaluate the run against manual labels in 11-class and 4-class space.
4. Inspect logs and distributions for the main failure modes.
5. Write a dated report under `new/reports/`.

## Commands

### 1. Sync remote artifacts

Prefer the existing sync script, overriding the host explicitly:

```bash
REMOTE_HOST="${host:-kiwi}" bash /Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/sync_kiwi_logs.sh --once
```

If the script is unavailable or unsuitable, fall back to direct `rsync` for:

- `annotated_samples_auto.csv`
- `annotation.log`
- `inference_process.jsonl`

### 2. Refresh local top-level CSV

Copy the synced CSV from:

- `new/annotation_tool/kiwi_logs/annotated_samples_auto.csv`

to:

- `new/annotation_tool/annotated_samples_auto.csv`

### 3. Evaluate against manual labels

Compute at least:

- valid prediction count
- missing sample IDs
- 11-class accuracy
- 11-class macro F1 and kappa when feasible
- 4-class accuracy
- 4-class macro F1 and kappa when feasible
- predicted distribution
- major confusion directions

If CSV and JSONL disagree, use the latest completed run in
`inference_process.jsonl` as the evaluation truth and call that out explicitly.

### 4. Analyze likely causes

Inspect:

- `kiwi_logs/inference_process.jsonl`
- `kiwi_logs/annotation.log`
- class distribution and stage-level choices

Explain not only what failed, but where the error seems to originate:

- VLM raw bias
- prompt routing bias
- summary misread
- graph rescue dependency
- stale CSV
- missing videos

### 5. Write the report

Create a dated report in `new/reports/` with a filename like:

- `YYYYMMDD_自動アノテーション人手比較レポート_<suffix>.md`

Use a descriptive suffix based on the actual run, not a generic one.

## Verification

- Confirm synced timestamps changed in `new/annotation_tool/kiwi_logs/`.
- Confirm `new/annotation_tool/annotated_samples_auto.csv` matches the synced
  CSV unless JSONL must override it.
- Confirm the report references the actual files used.
- Confirm the report includes both metrics and interpretation.

## Summary

Return:

```md
## Result
- **Action**: synced latest annotation artifacts, compared against manual labels, and wrote a report
- **Status**: success | partial | failed
- **Details**: valid sample count, key metrics, report path
```
