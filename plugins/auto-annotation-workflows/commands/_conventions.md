---
description: Repo-specific conventions for the auto-annotation workflow commands.
---

# Command Conventions

These commands are specialized for `/Users/rimpeihata/Desktop/auto-annotation-research`.

## Repo Defaults

- Default remote host is `kiwi`.
- Use `kiwi-rmt` only when the user explicitly asks or the run happened there.
- Remote annotation base path:
  - `~/workspace/auto-annotation-research/new/annotation_tool`
- Local synced log directory:
  - `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs`
- Manual labels:
  - `/Users/rimpeihata/Desktop/auto-annotation-research/new/results/annotated_samples_manual.0121.csv`
- Optional 4-class helper:
  - `/Users/rimpeihata/Desktop/auto-annotation-research/new/results/annotated_samples_manual.0121_macro4.csv`
- Report output directory:
  - `/Users/rimpeihata/Desktop/auto-annotation-research/new/reports`

## Evaluation Rules

- Report both 11-class and 4-class metrics when possible.
- If CSV and JSONL disagree, treat the latest `inference_process.jsonl` run as the source of truth and explicitly say so.
- Always include:
  - valid prediction count
  - missing sample IDs
  - class distribution
  - primary error modes
  - concrete next-step improvements

## Visualization Rules

- Default representative samples are `1, 7, 2, 4, 47` unless the user specifies others.
- Save only actual model inputs unless the user explicitly asks for a collage.
- For the current Qwen3-VL setup, that means:
  - `frame_1.png` to `frame_4.png`
  - `trajectory_summary_topdown.png`
  - `trajectory_summary_normalized.png`

## Implementation Rules

- Before code edits, read the latest relevant comparison report and the latest run logs.
- Prefer one focused improvement per change-set.
- If the proposed improvement mainly changes prompts, first verify there is literature support and record the basis in a report.
- Run at least targeted verification (`py_compile`, unit tests, or a tight smoke test) after edits.
- Commit meaningful code changes before moving to the next major direction.

## Research Rules

- For literature-backed proposals, prefer primary sources and cite them.
- Clearly separate:
  - findings from this repository's experiments
  - inferences from logs or code
  - claims supported by literature
- Proposal reports must end with an implementation order, not only ideas.
