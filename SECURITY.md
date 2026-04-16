# Security Policy

## Supported versions

Only the latest tagged release receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

Please report security issues **privately** via GitHub's private vulnerability
reporting (the "Report a vulnerability" button under the **Security** tab of
this repository), or by emailing `peter@aetheron.com`.

We aim to:
- Acknowledge the report within **3 business days**.
- Share our assessment and a remediation plan within **10 business days**.
- Publish a fix and advisory once a patched release is available.

Please do **not** open a public issue for a suspected vulnerability.

## Scope

In scope:
- Paths that could leak a user's `ANTHROPIC_API_KEY`, other credentials, or
  environment variables from `.env` through the training or evaluation code.
- Code-injection or path-traversal bugs in the data-generation, training,
  evaluation, or scripting layers.
- Vulnerabilities in the dependency graph declared by `pyproject.toml`
  that are reachable from first-party code.

Out of scope:
- Model behaviour, including confabulation, value-misalignment, or any
  output produced by the fine-tuned model. Neo-Logos is a research model;
  see the "Known Limitations" section of the README.
- Issues in third-party services (Anthropic API, HuggingFace, llama.cpp)
  that we integrate with but do not control.
- Reproducibility gaps that require retraining. File those as regular
  issues.

## Handling of training data

The training corpus is synthesised via the Anthropic Batch API. The
pipeline runs a decontamination pass that scrubs names listed in the
`CREATOR_NAMES` environment variable from generated text; operators
shipping derivatives should review the decontamination output before
release.
