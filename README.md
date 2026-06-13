# reference — Privacy-Preserving Inference Framework

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

| File | Function/Line | Role |
|---|---|---|
| `protocol/sshare.py:10-15` | `gen_add_share()` | Generates random additive mask `r` with zero-mean uniform distribution ±8 |
| `protocol/ptobase.py:48-63` | `_gen_add_share_()` | Creates or reuses additive share for a given shape |
| `protocol/scale.py:40-47` | `ProtocolClient.send_online()` | **Masking**: sends `data - r` to server (line 42) |
| `protocol/scale.py:49-59` | `ProtocolClient.recv_online()` | **Unmasking**: adds cached `self.pre` to server response (line 54) |
| `protocol/scale.py:21-23` | `ProtocolClient.setup()` | Generates `self.r` for this layer |
| `protocol/scale.py:25-38` | `ProtocolClient.send_offline()` / `recv_offline()` | Sends `r` to server, caches result as `self.pre` |
| `layer/base.py:16-47` | `LayerClient` | Wires `ProtocolClient` into the per-layer pipeline |
| `layer/conv.py:11-13` | `ConvClient` | Inherits LayerClient which calls `self.protocol.send_online(xm)` → server sees only masked data |
| `layer/fc.py:10-12` | `FcClient` | Same pattern |
| `layer/relu.py:10-21` | `ReLUClient.online()` | **Local processing**: data never leaves client |
| `layer/flatten.py:10-21` | `FlattenClient.online()` | **Local processing**: data never leaves client |
| `layer/softmax.py:10-21` | `SoftmaxClient.online()` | **Local processing**: data never leaves client |

### Server privacy — multiplicative blinding + additive offset (`m`, `s`)

| File | Function/Line | Role |
|---|---|---|
| `protocol/sshare.py:18-24` | `gen_mul_share()` | Generates positive random multiplier `m ∈ [ε, 16)` per output element (line 24) |
| `protocol/scale.py:69-79` | `ProtocolServer.setup()` | Generates `s` (additive offset, line 78) and `m` (multiplicative scaling, line 79) |
| `protocol/scale.py:110-118` | `ProtocolServer.send_offline()` | Blinds: `data *= self.m; data += self.s` (lines 112-113) — client never sees bare `W·r` |
| `protocol/scale.py:135-146` | `ProtocolServer.send_online()` | Blinds: `data *= self.m; data -= self.s` (lines 140-141) — client never sees bare `W·(x-r)` |
| `protocol/scale.py:98-108` | `ProtocolServer.recv_offline()` | Divides received mask by `mlast`: `data /= self.mlast` (line 104) |
| `protocol/scale.py:120-133` | `ProtocolServer.recv_online()` | Divides received data by `mlast`: `data /= self.mlast` (line 128) |
| `layer/base.py:76-81` | `LayerServer.run_layer_offline()` | Removes bias during offline computation — bias only applied online, mixed with `m` and `s` |
| `layer/conv.py:21-31` | `ConvServer.offline()` | Runs `W·r` (no bias), result immediately blinded with `m`, `s` |
| `layer/conv.py:33-45` | `ConvServer.online()` | Runs `W·(x-r)` (with bias), result immediately blinded with `m`, `s` |
| `layer/fc.py:19-41` | `FcServer.offline()` / `online()` | Same pattern as ConvServer |
| `layer/avgpool.py:21-43` | `AvgPoolServer` | Same pattern (linear operation, no learnable params, but maintains protocol chain) |

### Server privacy (extra) — shuffle and noise

| File | Function/Line | Mechanism |
|---|---|---|
| `protocol/shuffle.py:74-97` | `ProtocolServer.setup()` | Generates random permutation `p = torch.randperm(n)` (line 94) |
| `protocol/shuffle.py:127-132` | `ProtocolServer.shuffle_output()` | Permutes output: `data.ravel()[self.p].reshape(self.oshape)` (line 131) |
| `protocol/shuffle.py:147-156` | `ProtocolServer.send_offline()` | Applies shuffle: `data = shuffle_output(data)` (line 150) |
| `protocol/shuffle.py:175-188` | `ProtocolServer.send_online()` | Applies shuffle: `data = shuffle_output(data)` (line 182) |
| `protocol/shuffle.py:120-125` | `ProtocolServer.deshuflle_input()` | Unshuffles incoming data using previous layer's permutation (line 124) |
| `protocol/noise.py:51-68` | `ProtocolServer.send_online()` | Injects Gaussian noise before applying blinding: `data += N(0, σ²)` (line 64) |

### Both parties — efficiency and infrastructure

| File | Function/Line | Role |
|---|---|---|
| `system/client.py:23-37` | `Client.offline()` | Runs all layers' offline prep in sequence (line 24-32) |
| `system/client.py:34-37` | `Client.online()` | Runs all layers' online pass in sequence (line 34-36) |
| `system/server.py:26-41` | `Server.offline()` | Runs all layers' offline prep in sequence (line 30-41) |
| `system/server.py:43-48` | `Server.online()` | Runs all layers' online pass in sequence (line 43-48) |
| `protocol/ptobase.py:106-124` | `ProBaseClient.basic_send/recv_offline()` | Optional HE encryption for offline data in transit |
| `comm/he.py:21-50` | `send/recv_he_matrix()` | Per-ciphertext HE serialization for encrypted offline messages |
| `protocol/plaintext.py` | All | No masking at all — benchmarking only, protects neither party |

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

## Relationship to ppvas2

This codebase is the **reference implementation** for the ppvas2 project (privacy-preserving video analytics v2). While this reference demonstrates the core secret-sharing protocol on **static image inference**, ppvas2 extends the same ideas to **video streams** with incremental processing:

- Frame differencing to skip unchanged regions
- Per-block extraction and masking (not full-frame)
- Batching of per-layer blocks via pseudo-frames
- Event-driven client loop with diff-propagation across layers
- Multi-threaded server with block buffer / result buffer / scheduling

The protocol (`scale` with additive + multiplicative blinding), layer classification (linear→server, non-linear→client), and two-phase execution pattern are identical.
