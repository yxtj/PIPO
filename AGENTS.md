# PIPO

Privacy-preserving CNN inference (secret sharing + multiplicative blinding, ICDCS 2024). Read `README.md` first for the protocol walkthrough; this file only covers what the README gets wrong or omits.

## Status: 2.0 in development

Mid-transition to 2.0 (more modules + encryption methods): the tree is a **large uncommitted reorganization** — `src/`, `models/`, `security/`, `tests/`, `train/`, `config.toml` are untracked and the README lags the code. Old `PIPO_USE_HE` env var is gone; HE is config-driven (`use_he` / `--use-he`).

## Root layout

Framework code lives in `src/`; the rest is split across root-level dirs — check both before assuming a module is missing:

- `src/` — the framework: entrypoint `src/main.py` + packages `comm/ common/ heutil/ layer/ layer_basic/ model/ plot/ protocol/ system/ tests/`.
- `models/` (root) — model definitions/registry (`registry.py`, minionn, resnet, vgg, ...). Don't confuse with **`src/model/`** (singular) = `DagModel` graph infra, which stays under `src/`.
- `security/` (root) — standalone model-security analysis (attacks, DP/permutation bounds); imports `src.model`, `models`, `train`.
- `train/` (root) — training scripts (`minionn.py`, `resnet.py`, `util.py`), not `src/train/`.
- `tests/` (root) — system-vs-local comparison harness; `src/tests/` is a separate older smoke-test set.
- `poc/` (root) — DP-noise probes (`noise_relu.py`, `noisy_model.py`); gitignored (`/poc/`).
- `data` — junction → `D:\Data` (CIFAR10/CIFAR100/COCO); scripts use `data/...` from the repo root, re-create per machine.
- Gitignored / never committed: `pretrained/`, `log/`, `reference/`, `poc/`, `pypath.bat`, `opencode.jsonc`, `data`.
- Other root files: `config.toml` (effective config, read by both roles), `ckks-parameter.txt` (HE parameter reference), `pypath.bat` (conda env `privacy` + PYTHONPATH on this machine).

## Run

```bash
# Python >= 3.11 (uses tomllib). On this machine, run inside conda env
# "privacy" — see pypath.bat — from the repo root.
python -m src.main server            # terminal 1
python -m src.main client --verify   # terminal 2
```

One entry point serves both roles; `config.toml` + CLI flags (flags win) pick the model (`models/registry.py` by name), protocol, weights, host/port, HE.

## Config & protocol dispatch quirk

- `src/common/config.py` loads `config.toml` and applies CLI overrides; the effective config must be committed via `set_config()` before dependent imports.
- `src/protocol/__init__.py` **dynamically imports the selected protocol (`plaintext|scale|shuffle|noise`) at import time from the config** — never import `src.protocol.scale` etc. directly; adding a protocol means also updating `AVAILABLE_PROTOCOL` and `src/main.py`'s `_AVAILABLE_PROTOCOL`.
- Pyfhel (HE) is imported lazily; `src/main.py` hard-codes the CKKS context (`n=2**13, scale=2**30, qi_sizes=[30]*5`).

## Tests

- Root `tests/`: spawn `python -m src.main server` as a subprocess, run the client in-process on the same weights, compare with local `model(x)`. Protocol is an extra arg: `python tests/poc.py scale`. Helpers: `tests/session.py` `run_compare`/`check_threshold`; `plaintext` must match ≈0 error, masked protocols allow FP-error bounds.
- `src/tests/`: separate older smoke-test set — standalone scripts (e.g. `python src/tests/crypto/rsa.py`); `src/tests/network/{client,server}.py` exercise AES-CBC (`cryptography`) + Pyfhel.

## Crypto / encryption (2.0 expansion area)

- `src/heutil/` — Pyfhel CKKS helpers; `src/heutil/shen/` — alternate HE impl + `beaver_OT.py`; `src/comm/he.py` serializes `PyCtxt` over the wire.
- `src/comm/rsa.py` (pycryptodome RSA, `encrypt_big` for long payloads) and `src/comm/ot.py` (Oblivious Transfer) back the offline protocol.
- New encryption methods being added in 2.0: wire them through the config/CLI and `comm/` serialization; keep Pyfhel/cryptography imports lazy (optional deps).

## Model / layer architecture

- All models are `DagModel` (`src/model/dag_model.py`), an `nn.Sequential` subclass with explicit `AddOp` / `ConcatOp` / `JumpOp` feature-map sources.
- `src/system/util.py` maps each `nn.Module` type to a `LayerClient`/`LayerServer` pair (offline = precompute `W·r`; online = one masked round trip). Add new layer types there, not just to the model.

<!-- CODEGRAPH_START -->

## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.

<!-- CODEGRAPH_END -->