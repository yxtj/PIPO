# PIPO

**PIPO: Privacy-Preserving Convolutional Neural Network Inference with Plaintext Operations** (Zhou & Gao, ICDCS 2024)

https://ieeexplore.ieee.org/abstract/document/10630913

This project is the mplementationof the paper, hosting the full protocol, model, and networking code. It demonstrates the core secret-sharing protocol, the offline/online phase split, and the client–server layer architecture.

- [How it works](#how-it-works)
- [Two-phase execution](#two-phase-execution)
- [The scale protocol, step by step](#the-scale-protocol-step-by-step)
- [How to run](#how-to-run)
- [Protocol variants & configuration](#protocol-variants--configuration)
- [Example models](#example-models)
- [Directory structure](#directory-structure)
- [Layer classification](#layer-classification)
- [Key design decisions](#key-design-decisions)
- [Dependencies](#dependencies)
- [Citation](#citation)

## Overview

PIPO is a research framework for **client–server privacy-preserving inference (PPI)**. It protects **both** the client's input data and the server's model parameters using additive secret sharing and multiplicative blinding:

- **Client's data** `x` is never seen in plaintext by the server (additive masking `x → r, x-r`)
- **Server's model weights** `W` cannot be reconstructed by the client (multiplicative blinding `m` + additive offset `s`)

Linear operations (conv, fc, avg pool) run on the server under additive masking; non-linear operations (ReLU, softmax, flatten) happen locally on the client. Most protocol complexity goes toward server privacy.

## How it works

### Client privacy: additive secret sharing

Each layer input `x` is additively masked as `x → (r, x-r)`. The server receives only `x - r` and computes `W·(x - r)`. The client recovers the true result by adding the precomputed `W·r`:

```
W·(x - r) + W·r = W·x
```

Key files: `src/protocol/sshare.py:gen_add_share()`, `src/protocol/scale.py:ProtocolClient.send_online()` (`data - r`), `src/protocol/scale.py:ProtocolClient.recv_online()` (`data + pre`).

### Server privacy: multiplicative + additive blinding

The server blinds every tensor it sends back with a random positive multiplier `m` and a random additive offset `s`. The client sees:

```
offline:  W·r·m + s   → cached as `pre`
online:   W·(x-r/m)·m - s
combined: W·x·m
```

The client always recovers an output scaled by the unknown `m` — never the true `W·x`. This prevents the client from learning `W` through the input–output relationship. For the `shuffle` protocol, an element-wise permutation `p` is also applied, breaking spatial correspondence entirely.

Key files: `src/protocol/sshare.py:gen_mul_share()`, `src/protocol/scale.py:ProtocolServer.setup()` (generates `s`, `m`), `src/protocol/scale.py:ProtocolServer.send_offline()` (`data *= m; data += s`).

## Two-phase execution

Every inference session has exactly two phases:

| Phase       | When                           | What happens                                                                                                                        |
| ----------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Offline** | Once, before data is available | Client generates random masks `R_i` per layer, sends to server; server runs the masks through each linear layer; results are cached |
| **Online**  | Per inference                  | Client sends masked input, server computes, client unmasks and applies local non-linear ops                                         |

Both server and client must complete offline before online begins. The **offline phase** precomputes `W·r` for every layer before data arrives; the **online phase** then only needs to send `x - r` and unmask the result — one round trip per inference. The offline phase can optionally be encrypted with HE (`use_he = true` / `--use-he`) for protection against network eavesdroppers.

The protocol flow of each linear layer:

1. **Offline (data-independent):** client sends random mask `r` → server computes `W·r` (without bias) and applies `m`, `s` → client caches `W·r·m + s`
2. **Online (data-dependent):** client sends `x - r` → server computes `W·(x - r)` (with bias), blinds with `m`, `-s` → client unmasks using the cached value → recovers `W·x·m`

Key files: `src/system/client.py:Client.offline()`, `src/system/server.py:Server.offline()`, `src/comm/he.py`.

## The scale protocol, step by step

A complete run for a `Conv → ReLU → Linear` chain:

```
Offline:
  Client sends r₁ (additive mask for layer 1)
  Server computes W₁·r₁ (no bias), returns (W₁·r₁)·m₁ + s₁
  → Client caches pre₁ = W₁·r₁·m₁ + s₁

  Client sends r₂ (additive mask for layer 2)
  Server computes W₂·r₂/m₁ (no bias), returns (W₂·r₂/m₁)·m₂ + s₂
  → Client caches pre₂ = W₂·r₂·m₂/m₁ + s₂

Online (one inference):
  Client sends x - r₁
  Server receives, divides by 1 (first layer), computes W₁·(x - r₁) (WITH bias)
  Server returns W₁·(x - r₁)·m₁ - s₁
  Client: W₁·(x - r₁)·m₁ - s₁ + pre₁ = W₁·x·m₁ + bias·m₁   ← scaled by m₁, unknown

  ReLU: ReLU(W₁·x·m₁) = ReLU(W₁·x)·m₁   (m₁ > 0 preserves sign)
  Client sends ReLU(W₁·x)·m₁ - r₂
  Server receives, divides by m₁ → ReLU(W₁·x) - r₂/m₁
  Server computes W₂·(ReLU(W₁·x) - r₂/m₁), returns W₂·(...)·m₂ - s₂
  Client: W₂·(...)·m₂ - s₂ + pre₂ = W₂·ReLU(W₁·x)·m₂   ← scaled by m₂, unknown
```

The client's final output is always `f(x)·m_last` — the server never reveals plaintext model output, and the client never reveals plaintext input. The multiplicative mask `m` propagates through all local layers (ReLU preserves sign, Flatten reshapes, Softmax is applied client-side).

## How to run

One entry point serves both roles; the model and all launch parameters come
from `config.toml` / CLI arguments:

```bash
# terminal 1 - server (listens)
python -m src.main server

# terminal 2 - client (connects and verifies the result)
python -m src.main client --verify
```

Override anything through CLI flags (they win over `config.toml`):

```bash
python -m src.main server --model minionn --wfile pretrained/minionn.pt --protocol scale
python -m src.main client --model minionn --protocol scale --verify --n 2
```

Model training lives under `train/` (run from the repo root):

```bash
python train/resnet.py data_dir chkpt_dir [epochs] [dump_interval] [bs] [lr] [device] [model_version]
python train/minionn.py data_dir chkpt_dir [epochs] [batch_size] [dump_interval] [lr] [device]
```

## Tests

`tests/` (at the repo root) compares the **privacy-preserving system result**
(client→server run) with the **pure client mode** result — the local plaintext
`model(x)` computed in the test process:

```bash
python tests/poc.py                    # all poc-* models
python tests/minionn.py                # MiniONN (pretrained weights)
python tests/resnet.py                 # resnet-cifar-20
python tests/openpose.py body          # OpenPose body (pretrained weights)
python tests/run_all.py                # everything above
# protocol variant as extra arg: plaintext | scale | shuffle | noise
python tests/poc.py plaintext
```

Each script spawns the server (`python -m src.main server`) as a subprocess,
runs the client in-process with the same weights (`torch.manual_seed` or a
weights file), then reports

```
[scale] model poc-5: abs diff mean 8.9e-05 / max 1.9e-04, rel(diff) 3.2e-05, rel(norm) 1.4e-06, online 0.004s
```

where `rel(diff)` is the mean element-wise relative difference and `rel(norm)`
the mean absolute difference normalized by the typical output magnitude.
`plaintext` mode must reproduce the local result (≈0 error); for masked
protocols the residual is floating-point error accumulated across the protocol
chain (deeper models allow a looser bound per script).

## Protocol variants & configuration

Behavior is selected through `config.toml` (`[common]` section) or CLI flags:

| Option (`config.toml`) | CLI flag   | Values                                   | Default | Description                                    |
| ---------------------- | ---------- | ---------------------------------------- | ------- | ---------------------------------------------- |
| `protocol`             | `--protocol` | `plaintext`, `scale`, `shuffle`, `noise` | `scale` | Security protocol variant                      |
| `use_he`               | `--use-he` | `true`, `false`                          | `false` | Enable HE in the offline phase                 |
| `model`                | `--model`  | any name from `models/registry.py`       | `poc-1` | The model both roles must use identically      |
| `wfile`                | `--wfile`  | path                                     | (empty) | Pretrained weights file (state dict)           |
| `host`/`port`          | `--host`/`--port` | -                                 | `127.0.0.1:8100` | Connection endpoints                  |
| `n`                    | `--n`      | int                                      | `1`     | Number of online inference rounds              |
| `seed`                 | `--seed`   | int                                      | `0`     | RNG seed for random weights (shared)           |

```toml
# config.toml
[common]
host = "127.0.0.1"
port = 8100
model = "poc-1"
protocol = "scale"
use_he = false
```

```bash
# Plaintext (no masking, benchmarking only)
python -m src.main server --protocol plaintext

# Default scale protocol (protects both client and server)
python -m src.main server --protocol scale

# Enable HE in the offline phase (protects offline messages from eavesdroppers)
python -m src.main server --use-he
```

| Protocol          | Client privacy      | Server privacy                         | Mechanism                                                                              |
| ----------------- | ------------------- | -------------------------------------- | -------------------------------------------------------------------------------------- |
| Protocol          | Client privacy      | Server privacy                         | Mechanism                                                                              |
| ----------------- | ------------------- | -------------------------------------- | -------------------------------------------------------------------------------------- |
| `plaintext`       | None                | None                                   | Direct data transfer, no masking                                                       |
| `scale` (default) | ✅ Additive mask `r` | ✅ Multiplicative `m` + additive `s`    | `x → x-r` on client; `data → data·m ± s` on server                                     |
| `shuffle`         | ✅ Same as scale     | ✅ Scale + element-wise permutation `p` | Server permutes output elements before sending, client cannot map values to positions  |
| `noise`           | ✅ Same as scale     | ✅ Scale + Gaussian DP noise            | Server adds `N(0, σ²)` to output, prevents precise reconstruction via repeated queries |

## Example models

Models are resolved by name through `models/registry.py`; every entry
names a builder under `models/`:

| Registry name                      | Model                                      | Input shape     | Skip connections                              |
| ---------------------------------- | ------------------------------------------ | --------------- | --------------------------------------------- |
| `minionn`                          | MiniONN (small conv net for CIFAR)         | `(3, 32, 32)`   | No (`nn.Sequential`)                          |
| `resnet-18/34/50/101/152`          | ImageNet-style ResNet                      | `(3, 224, 224)` | Yes (Jump + Addition)                         |
| `resnet-cifar-20/32/44/56/110/152` | ResNet on CIFAR-100 (+versions 1–4)        | `(3, 32, 32)`   | Yes (Addition shortcuts via `DagModel`)       |
| `vgg-11/13/16/19`                  | VGG (ImageNet head)                        | `(3, 224, 224)` | No                                            |
| `openpose-body` / `openpose-hand`  | OpenPose pose estimation                   | `(3, 368, 368)` | Yes (Jump + Concatenation)                    |
| `poc-*`                            | Small custom models for prototyping        | Variable        | Yes (all DAG node types, incl. multi-branch)  |

The models are built as
[`DagModel`s](src/model/dag_model.py) — an
`nn.Sequential` subclass with an explicit feature-map source graph, so skip
connections (`AddOp`, `ConcatOp`, `JumpOp`) and multi-branch concat are
expressed directly in the model definition:

```python
from src.model.dag_model import AddOp, ConcatOp, DagModel, JumpOp

model = DagModel(
    nn.Conv2d(1, 5, 3),          # 0: FM[0] -> FM[1]
    nn.ReLU(),                   # 1: FM[1] -> FM[2]
    nn.Conv2d(5, 4, 3, 1, 1),    # 2: FM[2] -> FM[3]
    nn.ReLU(),                   # 3: FM[3] -> FM[4]
    nn.Conv2d(5, 6, 3, 1, 1),    # 4: FM[4] -> FM[5]
    nn.ReLU(),                   # 5: FM[5] -> FM[6]
    (ConcatOp(1), [-3, None]),   # 7: FM[3] + FM[6] -> FM[7]
)
```

## Directory structure

```
PIPO/
├── config.toml         # Default configuration (shared by client and server)
├── models/             # Neural-network model definitions
│   ├── registry.py     #   name -> (inshape, builder)
│   ├── resnet.py       #   ResNet builder (ImageNet + CIFAR)
│   ├── minionn.py      #   MiniONN builder
│   ├── openpose.py     #   OpenPose builder
│   ├── op_impl.py      #   OpenPose body/hand model internals
│   ├── vgg.py          #   VGG builder
│   └── poc.py          #   Small POC models map
├── train/              # Neural-network training code
│   ├── resnet.py       #   ResNet-CIFAR training loop
│   ├── minionn.py      #   MiniONN training loop
│   └── util.py         #   loader / train / checkpoint helpers
├── security/           # Standalone model-security analysis (attacks, bounds)
├── poc/                # Protocol-probing experiments (DP noise)
├── tests/              # Result-comparison tests (system vs pure-client local result)
│   ├── session.py      #   subprocess server + in-process client + diff metrics
│   ├── poc.py          #   all poc-* models
│   ├── minionn.py      #   MiniONN
│   ├── resnet.py       #   ResNet-CIFAR / VGG
│   ├── openpose.py     #   OpenPose body / hand
│   └── run_all.py      #   runs all of the above
├── src/
│   ├── main.py         # The single entry point (server|client)
│   ├── model/          # Model graph infrastructure
│   │   └── dag_model.py#   DagModel (+AddOp/ConcatOp/JumpOp Sh aps)
│   ├── common/         # Shared config / model IR helpers
│   │   ├── config.py   #   config.toml loading + CLI override machinery
│   │   ├── model_info.py   # static shape computation / structural IR
│   │   ├── model_info_io.py#
|   |   `-- types.py    #   ExecutionStep / ModelInfo dataclasses
│   ├── system/         # Wires layers + protocol + networking
│   │   ├── client.py   #   Client orchestration (setup → offline → online)
│   │   ├── server.py   #   Server orchestration (setup → offline → online)
│   │   ├── runner.py   #   Convenience: run_client() / run_server()
│   │   └── util.py     #   Maps nn.Module layer types to client/server layer classes
│   ├── layer/          # Layer abstractions (one per PyTorch layer type)
│   │   ├── base.py     #   LayerClient, LayerServer, LocalLayer*
│   │   ├── conv.py     #   ConvClient / ConvServer
│   │   ├── fc.py       #   FcClient / FcServer
│   │   ├── relu.py     #   ReLUClient / ReLUServer (local)
│   │   ├── maxpool.py  #   MaxPoolClient / MaxPoolServer (remote, Kronecker masks)
│   │   ├── avgpool.py  #   AvgPoolClient / AvgPoolServer (remote, linear)
│   │   ├── flatten.py  #   FlattenClient / FlattenServer (local)
│   │   ├── softmax.py  #   SoftmaxClient / SoftmaxServer (local)
│   │   ├── shortcut.py #   Addition / Concatenation / Jump (remote, DAG-aware)
│   │   `-- identity.py #   IdentityClient / IdentityServer (pass-through)
│   ├── layer_basic/    # Shared utilities
│   │   ├── layercommon.py # Base class for all layers
│   │   `-- stat.py     #   Timing/byte statistics dataclass
│   ├── protocol/       # Secret-sharing protocol implementations
│   │   ├── __init__.py #   dynamic dispatch on config `protocol`
│   │   ├── ptobase.py  #   Base client/server protocol classes
│   │   ├── sshare.py   #   Additive/multiplicative share generation
│   │   ├── plaintext.py#   No masking (benchmark)
│   │   ├── scale.py    #   Additive + multiplicative blinding (default)
│   │   ├── shuffle.py  #   Scale + element-wise shuffle
│   │   `-- noise.py    #   Scale + differential privacy noise
│   ├── comm/           # Network communication (raw TCP)
│   ├── heutil/         # Homomorphic-encryption helpers (Pyfhel)
│   ├── tests/          # Unit/integration tests
│   └── plot/           # Paper-figure generators (CSV-based)
├── pretrained/         # Pretrained weights (minionn, resnet, openpose)
└── reference/          # Third-party reference implementations
```

## Layer classification

Layers are split by where computation happens and what the computation is:

| Category               | Layers                              | Location | Computation                                                 |
| ---------------------- | ----------------------------------- | -------- | ----------------------------------------------------------- |
| **Remote, linear**     | Conv2d, Linear, AvgPool2d, Identity | Server   | `W·(x - r)` under additive mask                             |
| **Remote, non-linear** | MaxPool2d                           | Server   | Kronecker-product expanded mask for non-overlapping pooling |
| **Client, non-linear** | ReLU, Softmax, Flatten              | Client   | Applied directly on unblinded values                        |
| **Shortcut**           | Addition, Concatenation, Jump       | Server   | Buffered feature-merging driven by the `DagModel` source graph |

## Key design decisions

- **Raw TCP sockets** for communication (no ZeroMQ, gRPC, or HTTP). Each message is a 4-byte length header followed by the serialized payload.
- **`DagModel` carries the model graph**: a pure-Python `nn.Sequential` subclass where each layer declares its feature-map sources, so skip connections and multi-branch topologies are data, not hand-written buffers.
- **`config.toml` + CLI overrides** decide the protocol variant, HE usage and which registered model runs — `src/main.py` is the single entry point for both roles.
- **`protocol/__init__.py` dynamically imports** the selected protocol — never import `protocol.scale` or `protocol.shuffle` directly.
- **Statistics are captured per layer** via the `Stat` dataclass — bytes sent/received, computation/wait time, broken down by offline vs online phase.

## Dependencies

```bash
pip install torch torchvision Pyfhel pycryptodome scipy numpy opencv-python
```

- `Pyfhel` — HE support (optional, only if `PIPO_USE_HE=1`)
- `pycryptodome` — RSA encryption for Oblivious Transfer
- `opencv-python` — used by OpenPose example for image preprocessing

## Citation

If you use this code in your research, please cite:

[Zhou2024PIPO]
Zhou, T. & Gao, L. (2024). PIPO: Privacy-Preserving Convolutional Neural Network Inference with Plaintext Operations. In *2024 IEEE 44th International Conference on Distributed Computing Systems (ICDCS)* (pp. 1365–1376). IEEE.

```bibtex
@inproceedings{zhou2024pipo,
  title={PIPO: Privacy-Preserving Convolutional Neural Network Inference with Plaintext Operations},
  author={Zhou, Tian and Gao, Lixin},
  booktitle={2024 IEEE 44th International Conference on Distributed Computing Systems (ICDCS)},
  pages={1365--1376},
  year={2024},
  organization={IEEE}
}
```
