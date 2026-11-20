# AMAM evaluation snapshot

The `amam-eval` branch is based on upstream commit `7d57e4722777537a2fbe1cc181485c393dfacf68`, the environment generation used by AMAM. It intentionally does not adopt subsequent upstream restructuring. All 102 original tracked `src` files, including the 16 tasks and success evaluators, are preserved byte-for-byte.

Changes outside the task implementation provide dependency pins and a websocket policy entry point with head and wrist observations. Install using `pyproject.toml` and follow the upstream dataset/asset instructions. `python main.py --help` describes the standalone client. The AMAM memory-agent rollout controller is provided by AMAM separately.

The reported integrated16 epoch3 evaluation uses the official test episodes 0–49 for each of 16 tasks, policy seed 7, 16 executed actions per replan and a 1300-step rollout cap (not the builder's default 10000). Success is taken from the environment status; an Agent `finish` or `stop` does not count as success.

`amam_environment.json` records the frozen source hashes. Weights and runtime outputs are not part of this repository.
