---
description: Read the latest reports and logs, pick one high-signal improvement, implement it, verify it, and commit the change.
---

# implement-next-improvement

Implement the next focused improvement based on the latest evaluation evidence,
verify it, and commit it.

## Arguments

- `focus`: optional focus area such as `prompt`, `summary`, `graph`, `logging`,
  `sync`, `runtime`
- `report`: optional report path to prioritize
- `commit_message`: optional commit message override

## Preflight

1. Read the latest relevant evaluation report in `new/reports/`.
2. Read the latest synced logs in:
   - `new/annotation_tool/kiwi_logs/inference_process.jsonl`
   - `new/annotation_tool/kiwi_logs/annotation.log`
3. Check `git status --short` and note unrelated dirty files.
4. If the improvement is primarily a prompt change, first confirm there is
   literature support and record the basis in a new or updated report.

## Plan

1. Identify the single highest-signal bottleneck from the latest evidence.
2. Implement the narrowest meaningful code change that addresses it.
3. Run targeted verification.
4. Summarize what changed and commit it.

## Commands

### 1. Pick one focused change

Prefer one of:

- summary rendering change
- prompt routing change
- graph override rule
- runtime metadata/logging fix
- stale CSV / sync correction

Do not mix multiple speculative changes unless they are tightly coupled.

### 2. Ground prompt edits properly

If editing prompts:

- consult existing prompt rationale documents first
- survey literature before changing behavior-heavy prompt logic
- explicitly distinguish literature-backed changes from repo-only heuristics

### 3. Implement with repo conventions

- Use existing code paths rather than parallel prototypes.
- Preserve current input/output file contracts unless the change requires a
  migration.
- Do not commit generated CSVs or remote-only artifacts unless the user
  explicitly asks.

### 4. Verify

Run the smallest useful checks, such as:

- `python3 -m py_compile ...`
- targeted unit tests
- a small rendering smoke test

If a full evaluation run is needed, say so, but do not fabricate it.

### 5. Commit

Commit meaningful code changes after verification. Keep the commit scoped to
the focused improvement.

## Verification

- Confirm the targeted files changed as intended.
- Confirm the focused checks passed.
- Confirm the change is explained in plain language.
- Confirm a commit was created if code changed.

## Summary

Return:

```md
## Result
- **Action**: implemented one focused improvement
- **Status**: success | partial | failed
- **Details**: key files, verification run, commit SHA
```
