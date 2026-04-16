---
name: Bug report
about: Something isn't working — training crashes, generation fails, tests error, etc.
title: "[bug] "
labels: bug
---

## What happened

<!-- One or two sentences. What did you run, what did you expect, what did you get? -->

## Reproduction

```bash
# The exact command that produced the problem
```

## Environment

- **OS**: <!-- e.g., Ubuntu 24.04, WSL2, macOS 14 -->
- **GPU**: <!-- e.g., RTX 5090, A100 40GB, none -->
- **CUDA**: <!-- e.g., 12.8, N/A -->
- **Python**: <!-- `python --version` -->
- **Package version**: <!-- `pip show neo-logos-training | grep Version` — or commit hash if installed from source -->
- **Unsloth**: <!-- `pip show unsloth | grep Version` (if relevant to the failure) -->
- **torch**: <!-- `pip show torch | grep Version` (if relevant) -->

## Logs / error output

<details>
<summary>Full traceback</summary>

```
paste the full traceback or the last ~50 lines of the log here
```

</details>

## Additional context

<!-- Config changes you made, data quirks, anything else worth knowing. -->
