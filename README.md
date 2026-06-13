# PIPO

An efficient privacy-preserving network network inference framework.

PIPO is short for “Privacy-Preserving Inference with Plaintext Operations”

A research framework for **client-server privacy-preserving inference (PPI)** that protects **both** the client's input data and the server's model parameters. The framework uses additive secret sharing and multiplicative blinding to ensure that:

- **Client's data** `x` is never seen in plaintext by the server (additive masking `x → r, x-r`)
- **Server's model weights** `W` cannot be reconstructed by the client (multiplicative blinding `m` + additive offset `s`)

Linear operations (conv, fc, avg pool) happen on the server under additive masking; non-linear operations (ReLU, softmax, flatten) happen locally on the client. Most protocol complexity goes toward server privacy.

This codebase serves as a **reference implementation** for the ppvas2 privacy-preserving video analytics project. It demonstrates the core secret-sharing protocol, the offline/online phase split, and the client-server layer architecture.

## Core technique: dual-party privacy

### Client privacy: additive secret sharing

Each layer input `x` is additively masked as `x → (r, x-r)`. The server receives only `x - r` and computes `W·(x - r)`. The client recovers the true result by adding the precomputed `W·r`:

```
W·(x - r) + W·r = W·x
```

Key files: `protocol/sshare.py:gen_add_share()` (lines 10-15), `protocol/scale.py:ProtocolClient.send_online()` (line 42: `data - r`), `protocol/scale.py:ProtocolClient.recv_online()` (line 54: `data + pre`).

### Server privacy: multiplicative + additive blinding

The server blinds every tensor it sends back with a random positive multiplier `m` and a random additive offset `s`. The client sees:

```
offline:  W·r·m + s   → cached as `pre`
online:   W·(x-r/m)·m - s
combined: W·x·m
```

The client always recovers an output scaled by the unknown `m` — never the true `W·x`. This prevents the client from learning `W` through the input-output relationship. For the `shuffle` protocol, an element-wise permutation `p` is also applied, breaking spatial correspondence entirely.

Key files: `protocol/sshare.py:gen_mul_share()` (lines 18-24), `protocol/scale.py:ProtocolServer.setup()` (lines 78-79: generates `s`, `m`), `protocol/scale.py:ProtocolServer.send_offline()` (lines 112-113: `data *= m; data += s`).

### Efficiency: two-phase execution

The **offline phase** precomputes `W·r` for every layer before data arrives. The **online phase** then only needs to send `x - r` and unmask the result — one round trip per inference. The offline phase can also be encrypted with HE (`PIPO_USE_HE=1`) for protection against network eavesdroppers.

Key files: `system/client.py:Client.offline()`, `system/server.py:Server.offline()`, `comm/he.py`.

**Protocol flow for each linear layer:**

1. **Offline (data-independent):** Client sends random mask `r` → Server computes `W·r` (without bias, via `run_layer_offline()`), applies `m`, `s` → Client caches `W·r·m + s`
2. **Online (data-dependent):** Client sends `x - r` → Server computes `W·(x - r)` (with bias), blinds with `m`, `-s` → Client unmasks using cached value → recovers `W·x·m`

## Two-phase execution

Every inference session has exactly two phases:

| Phase | When | What happens |
|---|---|---|
| **Offline** | Once, before data is available | Client generates random masks `R_i` per layer, sends to server; server runs the masks through each linear layer; results are cached |
| **Online** | Per inference | Client sends masked input, server computes, client unmasks and applies local non-linear ops |

Both server and client must complete offline before online begins.

## Protocol selection (via env vars)

| Variable | Values | Default | Description |
|---|---|---|---|
| `PIPO_PROTOCOL` | `plaintext`, `scale`, `shuffle`, `noise` | `scale` | Security protocol variant |
| `PIPO_USE_HE` | `0`, `1` | `0` | Enable homomorphic encryption in offline phase |

```bash
# Plaintext (no masking, benchmarking only)
set PIPO_PROTOCOL=plaintext

# Default scale protocol (protects both client and server)
set PIPO_PROTOCOL=scale

# Scale + element-wise shuffle (extra server privacy)
set PIPO_PROTOCOL=shuffle

# Scale + differential privacy noise (extra server privacy)
set PIPO_PROTOCOL=noise

# Enable HE in the offline phase (protects offline messages from eavesdroppers)
set PIPO_USE_HE=1
```

Protocols and their privacy guarantees:

| Protocol | Client privacy | Server privacy | Mechanism |
|---|---|---|---|
| `plaintext` | None | None | Direct data transfer, no masking |
| `scale` (default) | ✅ Additive mask `r` | ✅ Multiplicative `m` + additive `s` | `x → x-r` on client; `data → data·m ± s` on server |
| `shuffle` | ✅ Same as scale | ✅ Scale + element-wise permutation `p` | Server permutes output elements before sending, client cannot map values to positions |
| `noise` | ✅ Same as scale | ✅ Scale + Gaussian DP noise | Server adds `N(0, σ²)` to output, prevents precise reconstruction via repeated queries |

## Privacy attribution by component

### Client privacy — additive secret sharing (`x → r, x-r`)

### Server privacy — multiplicative blinding + additive offset (`m`, `s`)

### Server privacy (extra) — shuffle and noise

### Both parties — efficiency and infrastructure

### How the full protocol unwinds (scale protocol)

Step by step for a `Conv → ReLU → Linear` chain:

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

## Example models

| Example | Model | Input shape | Skip connections |
|---|---|---|---|
| `example/resnet.py` | ResNet-20/32/44/56/110/152 on CIFAR-10/100 | `(3, 32, 32)` | Yes (Addition shortcuts via `te.SequentialShortcut`) |
| `example/minionn.py` | MiniONN (small conv net for CIFAR) | `(3, 32, 32)` | No (`nn.Sequential`) |
| `example/openpose.py` | OpenPose body/hand pose estimation | `(3, 368, 368)` | Yes (Jump + Concatenation) |
| `example/poc.py` | Small custom models for prototyping | Variable | Yes (all shortcut types) |

## Directory structure

```
reference/
├── example/           # Entry points (one per model)
│   ├── poc.py         #   Proof-of-concept with small models
│   ├── resnet.py      #   ResNet on CIFAR
│   ├── minionn.py     #   MiniONN
│   └── openpose.py    #   OpenPose body/hand
├── system/            # Wires layers + protocol + networking
│   ├── client.py      #   Client orchestration (setup → offline → online)
│   ├── server.py      #   Server orchestration (setup → offline → online)
│   ├── runner.py      #   Convenience: run_client() / run_server()
│   └── util.py        #   Maps nn.Module layer types to client/server layer classes
├── layer/             # Layer abstractions (one per PyTorch layer type)
│   ├── base.py        #   LayerClient, LayerServer, LocalLayer*
│   ├── conv.py        #   ConvClient / ConvServer
│   ├── fc.py          #   FcClient / FcServer
│   ├── relu.py        #   ReLUClient / ReLUServer (local)
│   ├── maxpool.py     #   MaxPoolClient / MaxPoolServer (remote, Kronecker masks)
│   ├── avgpool.py     #   AvgPoolClient / AvgPoolServer (remote, linear)
│   ├── flatten.py     #   FlattenClient / FlattenServer (local)
│   ├── softmax.py     #   SoftmaxClient / SoftmaxServer (local)
│   ├── shortcut.py    #   Addition / Concatenation / Jump (remote)
│   └── identity.py    #   IdentityClient / IdentityServer (pass-through)
├── protocol/          # Secret-sharing protocol implementations
│   ├── ptobase.py     #   Base client/server protocol classes
│   ├── sshare.py      #   Additive/multiplicative share generation
│   ├── plaintext.py   #   No masking (benchmark)
│   ├── scale.py       #   Additive + multiplicative blinding (default)
│   ├── shuffle.py     #   Scale + element-wise shuffle
│   └── noise.py       #   Scale + differential privacy noise
├── comm/              # Network communication (raw TCP)
│   ├── basic.py       #   Chunk send/recv, shape serialization
│   ├── tensor.py      #   PyTorch tensor serialization
│   ├── ndarray.py     #   NumPy array serialization
│   ├── he.py          #   Pyfhel ciphertext serialization
│   ├── ot.py          #   1-of-2 Oblivious Transfer (RSA-based)
│   └── rsa.py         #   RSA encryption wrapper
├── model/             # Model definitions
│   ├── resnet.py      #   ResNet builder (resnet20..resnet152)
│   ├── minionn.py     #   MiniONN builder
│   ├── openpose.py    #   OpenPose builder
│   ├── op_impl.py     #   OpenPose body/hand model internals
│   ├── vgg.py         #   VGG builder
│   └── poc.py         #   Small POC models map
├── layer_basic/       # Shared utilities
│   ├── layercommon.py #   Base class for all layers
│   └── stat.py        #   Timing/byte statistics dataclass
├── torch_extension/   # Custom PyTorch module extensions (pure Python)
│   ├── seqsc.py       #   SequentialShortcut (nn.Sequential + skip connections)
│   └── shortcut.py    #   ShortCut, Jump, Addition, Concatenation modules
└── setting.py         # Reads env vars PIPO_PROTOCOL, PIPO_USE_HE
```

## Layer classification

Layers are split by where computation happens and what the computation is:

| Category | Layers | Location | Computation |
|---|---|---|---|
| **Remote, linear** | Conv2d, Linear, AvgPool2d, Identity | Server | `W·(x - r)` under additive mask |
| **Remote, non-linear** | MaxPool2d | Server | Kronecker-product expanded mask for non-overlapping pooling |
| **Client, non-linear** | ReLU, Softmax, Flatten | Client | Applied directly on unblinded values |
| **Shortcut** | Addition, Concatenation, Jump | Server | Buffered feature-merging via te.SequentialShortcut |

## Key design decisions

- **Raw TCP sockets** for communication (no ZeroMQ, gRPC, or HTTP). Each message is a 4-byte length header followed by the serialized payload.
- **`torch_extension/` is pure Python** — no C++/CUDA build step needed for this reference implementation. The package provides `SequentialShortcut` (a subclass of `nn.Sequential`) and the shortcut layer types (`Addition`, `Concatenation`, `Jump`).
- **`setting.py` reads env vars at import time** — module-level side effects. Always use `from setting import USE_HE, PROTOCOL` rather than importing protocol modules directly.
- **`protocol/__init__.py` dynamically imports** the selected protocol — never import `protocol.scale` or `protocol.shuffle` directly.
- **Statistics are captured per layer** via the `Stat` dataclass — bytes sent/received, computation/wait time, broken down by offline vs online phase.

## Dependencies

```bash
pip install torch torchvision Pyfhel pycryptodome scipy numpy opencv-python
```

- `Pyfhel` — HE support (optional, only if `PIPO_USE_HE=1`)
- `pycryptodome` — RSA encryption for Oblivious Transfer
- `opencv-python` — used by OpenPose example for image preprocessing
