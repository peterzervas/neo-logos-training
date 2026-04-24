# End-To-End Training Smoke Tests

These scripts are GPU-oriented smoke tests for the SFT -> DPO -> chat path.
They are intentionally not part of default CI because they create large model
artifacts under `e2e_test/artifacts/` and require local accelerator support.

Run them manually from the repository root after setting up the training
environment:

```bash
python e2e_test/sft.py
python e2e_test/dpo.py
python e2e_test/chat_multi.py
```

Generated artifacts and logs are ignored by git.
