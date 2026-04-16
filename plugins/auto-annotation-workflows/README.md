# Auto Annotation Workflows

Repo-local Codex plugin for the repeated workflows in this research repository.

Available slash commands:

- `/sync-annotation-report`
- `/render-trajectory-inputs`
- `/implement-next-improvement`
- `/investigate-and-propose`

Each command assumes this repository layout and the current research conventions:

- default remote host is `kiwi`
- synced artifacts land in `new/annotation_tool/kiwi_logs/`
- manual labels live in `new/results/annotated_samples_manual.0121.csv`
- 4-class evaluation should be reported alongside 11-class evaluation whenever possible
- prompt-only changes must be grounded in literature before implementation
