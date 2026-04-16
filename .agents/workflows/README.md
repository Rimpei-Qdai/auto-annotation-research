# Project-Local Workflows

This repository defines reusable Codex workflows under
`/Users/rimpeihata/Desktop/auto-annotation-research/.agents/workflows`.

These are not guaranteed to appear as native slash commands in Codex Desktop.
Instead, treat them as project-local command definitions similar to spec-kit
workflow files.

## How To Use

In chat, call them explicitly, for example:

- `workflow sync-annotation-report を実行して`
- `workflow render-trajectory-inputs を sample_ids=1,7,2,4,47 で実行して`
- `workflow implement-next-improvement を focus=prompt で実行して`
- `workflow investigate-and-propose を focus=vlm-trajectory-reading で実行して`

Or inspect them locally with:

```bash
bash ./scripts/codex-workflow list
bash ./scripts/codex-workflow show sync-annotation-report
```

## Available Workflows

- `sync-annotation-report`
- `render-trajectory-inputs`
- `implement-next-improvement`
- `investigate-and-propose`

## Conventions

Common repo assumptions and guardrails live in:

- `/Users/rimpeihata/Desktop/auto-annotation-research/.agents/workflows/_conventions.md`
