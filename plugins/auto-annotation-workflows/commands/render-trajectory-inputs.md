---
description: Generate and save representative current model inputs for selected samples, including raw frames and trajectory summaries.
---

# /render-trajectory-inputs

Render the current annotation inputs for representative samples so the visual prompt design can be inspected directly.

## Arguments

- `sample_ids`: comma-separated sample IDs; default `1,7,2,4,47`
- `label_names`: optional human-readable names for directory labels
- `output_tag`: optional directory suffix
- `save_mode`: `inputs-only` (default) or `with-collage`

## Preflight

1. Confirm the current repo is `/Users/rimpeihata/Desktop/auto-annotation-research`.
2. Confirm the current code path is the one being inspected.
3. Confirm videos and trajectory generation paths exist locally under `new/filterd_video/`.
4. Default to `sample_ids=1,7,2,4,47` unless the user specifies otherwise.

## Plan

1. Load the selected samples from the annotation dataset.
2. Generate the current visual inputs exactly as the model sees them.
3. Save them to a dated report directory with an `index.csv`.
4. Show representative output paths and, when helpful, render a few example images in chat.

## Commands

### 1. Use the current model input path

Use the same code path that builds current model inputs in:

- `new/annotation_tool/heron_model_with_trajectory.py`
- `new/annotation_tool/visual_prompting.py`

Do not create an ad hoc alternative rendering path.

### 2. Save actual inputs

For each sample, save:

- `frame_1.png`
- `frame_2.png`
- `frame_3.png`
- `frame_4.png`
- `trajectory_summary_topdown.png`
- `trajectory_summary_normalized.png`

Do not generate `contact_sheet.png` unless `save_mode=with-collage` or the user explicitly asks.

### 3. Save under a dated directory

Create a directory under `new/reports/` like:

- `YYYYMMDD_visual_prompt_examples_<tag>`

and an `index.csv` with:

- sample ID
- label or alias
- source video path
- chosen frame indices when available
- output directory
- summary image paths

## Verification

- Confirm the saved images match the current code path, not an old overlay style.
- Confirm image count per sample is correct.
- Confirm `index.csv` points to existing files.
- If a sample cannot be rendered, report why and continue with the rest.

## Summary

Return:

```md
## Result
- **Action**: rendered representative model inputs
- **Status**: success | partial | failed
- **Details**: sample count, output directory, any skipped samples
```

## Next Steps

- If the summaries look ambiguous, use `/investigate-and-propose` for a deeper redesign.
- If a concrete visual fix is obvious, use `/implement-next-improvement`.
