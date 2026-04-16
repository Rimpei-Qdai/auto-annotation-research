---
description: Audit the current state, survey literature, and write a literature-backed proposal for a deeper improvement path.
---

# /investigate-and-propose

Understand the current failure mode, survey literature, and propose a deeper improvement plan with clear implementation order.

## Arguments

- `focus`: optional research focus such as `vlm-trajectory-reading`, `prompt-routing`, `graph-design`, `summary-design`, `macro-to-fine`
- `scope`: `narrow` or `broad`, default `broad`
- `host`: remote host for log sync if needed, default `kiwi`

## Preflight

1. Read the latest relevant reports from `new/reports/`.
2. Read the latest synced logs from `new/annotation_tool/kiwi_logs/`.
3. If the latest remote run has not been synced yet, sync it first from `host`, defaulting to `kiwi`.
4. Identify whether the main question is:
   - repo diagnosis
   - literature survey
   - architecture redesign
   - all three

## Plan

1. Summarize the current system state from code and latest runs.
2. Identify the highest-value unresolved bottleneck.
3. Search literature for approaches directly relevant to that bottleneck.
4. Propose a concrete improvement path with implementation phases.
5. Write a dated report with citations and a recommended order of work.

## Commands

### 1. Inspect current repo state

Read the code paths currently responsible for:

- model input generation
- prompt routing
- graph rescue
- evaluation and logging

### 2. Use literature intentionally

- Prefer primary sources over secondary summaries.
- For technical claims, cite the paper or official documentation directly.
- Clearly label which parts are:
  - observed in this repo
  - inferred from logs/code
  - supported by literature

### 3. Avoid superficial proposals

Do not stop at "improve prompts" or "try another model." Instead specify:

- what should change
- why it should help
- what evidence supports it
- how to test it
- what not to change yet

### 4. Write an implementation-oriented report

Create a dated report in `new/reports/` with:

- current-state summary
- key failure modes
- literature findings
- proposed architecture or experiment plan
- phased implementation order
- evaluation plan

## Verification

- Confirm the report cites actual sources.
- Confirm the proposal includes implementation order, not only high-level ideas.
- Confirm the report distinguishes evidence from speculation.

## Summary

Return:

```md
## Result
- **Action**: investigated current state and wrote a literature-backed proposal
- **Status**: success | partial | failed
- **Details**: focus area, sources used, report path
```

## Next Steps

- If the proposal identifies a concrete next code change, use `/implement-next-improvement`.
- If a new run is needed first, use `/sync-annotation-report` after it completes.
