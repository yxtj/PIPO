# PIPO

Privacy-preserving CNN inference (secret sharing + multiplicative blinding, ICDCS 2024; see `README.md` for the protocol walkthrough). Read the README first for architecture; this file only covers what the README gets wrong or omits.

## Status: 2.0 in development

The repo is mid-transition to 2.0 (more modules + more encryption methods). The working tree has a **large uncommitted reorganization**: old top-level `comm/`, `layer/`, `protocol/`, `heutil/`, `system/`, `security/` dirs were deleted and re-added under `src/`; `config.toml`, `tests/`, `train/`, `src/` are untracked. Watch for stale claims in the README:

- Training scripts live at root **`train/`** (`python train/minionn.py data_dir chkpt_dir ...`), not `src/train/` as the README's tree says.
- Old `PIPO_USE_HE` env var is gone; HE is config-driven (`use_he` / `--use-he`).
- Everything runs from the repo root (`python -m src.main ...`).

## Run

```bash
# Python >= 3.11 required (uses tomllib). On this machine, pypath.bat
# activates conda env "privacy" and sets PYTHONPATH to D:\Code\PIPO.
# or use `D:\miniconda3\envs\privacy\python.exe`  as the python entry.
python -m src.main server            # terminal 1
python -m src.main client --verify   # terminal 2
```

One entry point serves both roles; `config.toml` + CLI flags (flags win) pick the model, protocol, weights, host/port, HE. Models resolve by name through `src/models/registry.py`.

## Config & protocol dispatch quirk

- `src/common/config.py` loads `config.toml` and applies CLI overrides; the effective config must be committed via `set_config()` before dependent imports.
- `src/protocol/__init__.py` **dynamically imports the selected protocol (`plaintext|scale|shuffle|noise`) at import time based on the config** — never import `src.protocol.scale`, etc. directly, and don't add a protocol without updating `AVAILABLE_PROTOCOL` and `src/main.py`'s `_AVAILABLE_PROTOCOL`.
- Pyfhel (HE) is imported lazily everywhere; `src/main.py` hard-codes the CKKS context (`n=2**13, scale=2**30, qi_sizes=[30]*5`). `ckks-parameter.txt` at the root is the parameter reference.

## Tests

- Root `tests/` (README's "Tests" section): spawn `python -m src.main server` as a subprocess, run the client in-process against the same weights, compare with local `model(x)`. Protocol variant is an extra arg: `python tests/poc.py scale`. Helpers in `tests/session.py` (`run_compare`, `check_threshold`); `plaintext` must reproduce the local result (~0 error), masked protocols allow FP-error bounds.
- `src/tests/` is a different, older smoke-test set — standalone scripts run directly (e.g. `python src/tests/crypto/rsa.py`). `src/tests/network/{client,server}.py` also exercise AES-CBC (`cryptography`) and Pyfhel example code.

## Crypto / encryption (2.0 expansion area)

- `src/heutil/` — Pyfhel CKKS helpers; `src/heutil/shen/` — an alternate HE implementation plus `beaver_OT.py`. `src/comm/he.py` serializes `PyCtxt` over the wire.
- `src/comm/rsa.py` (pycryptodome RSA, `encrypt_big` for long payloads) and `src/comm/ot.py` (Oblivious Transfer) back the offline protocol.
- New encryption methods are being added in 2.0 — if you add one, wire it through the config/CLI and the corresponding `comm/` serialization, keep Pyfhel/cryptography imports lazy (they're optional).

## Model / layer architecture

- All models are `DagModel` (`src/model/dag_model.py`), a `nn.Sequential` subclass where skip connections are explicit `AddOp` / `ConcatOp` / `JumpOp` feature-map sources.
- `src/system/util.py` maps each `nn.Module` type to a `LayerClient`/`LayerServer` pair (offline = precompute `W·r`; online = masked inference, one round trip). Add a new layer type to that mapper, not just to the model.

<!-- CODEGRAPH_START -->

## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.

<!-- CODEGRAPH_END -->
