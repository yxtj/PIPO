# -*- coding: utf-8 -*-
"""Test-session helpers: launch the server as a subprocess (mirroring the old
`example/` scripts) and run the client in this process, then compare the
system result with the plaintext local result.

Under every protocol the last non-local layer's multiplicative mask is 1, so
the protocol result is directly comparable with `model(x)`; the difference
measures the protocol's numerical error.
"""

import subprocess
import sys
import tempfile
import time
import socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models import registry


def run_compare(model_name, protocol='scale', n=1, seed=0, wfile=None, device='cpu'):
    """Start `python -m src.main server` for the model, run the client here.

    Returns a dict with result / reference / diff metrics / timings.
    """
    torch.manual_seed(seed)
    inshape, model = registry.load(model_name, seed=seed, wfile=wfile, device=device)
    data = torch.rand(1, *inshape)
    if device.startswith('cuda') and torch.cuda.is_available():
        data = data.to(device)

    port = _free_port()
    # plain files: the server log must never fill an unserviced pipe
    out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
    args = [sys.executable, '-m', 'src.main', 'server',
            '--model', model_name, '--n', str(n), '--seed', str(seed),
            '--protocol', protocol, '--port', str(port)]
    if wfile:
        args += ['--wfile', str(wfile)]
    server = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT), stdout=out, stderr=err)
    try:
        from src.common.config import load_config, set_config
        from src.system import Client

        cfg = load_config()
        cfg.common.protocol = protocol
        cfg.common.use_he = False
        cfg.common.wfile = wfile or ''
        set_config(cfg)

        c = _connect(port)
        client = Client(c, model, inshape, None)
        client.offline()
        t0 = time.time()
        res = None
        for _ in range(n):
            with torch.no_grad():
                res = client.online(data)
        time_online = time.time() - t0
        c.close()
    finally:
        server.wait(timeout=120)  # raisable on hung/failed server
        server_err = _read(err)
        if server.returncode not in (0, None):
            raise RuntimeError('server run failed (code {}):\n{}'.format(server.returncode, server_err))
        out.close()

    with torch.no_grad():
        ref = model(data)
    diff = (res - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-12)
    ref_abs = ref.abs().mean().clamp_min(1e-12).item()
    return {
        'model': model_name,
        'protocol': protocol,
        'n': n,
        'result': res,
        'reference': ref,
        'diff': diff,
        'abs_diff_mean': diff.mean().item(),
        'abs_diff_max': diff.max().item(),
        'rel_diff_mean': rel.mean().item(),
        # relative error against the typical output magnitude (robust to
        # near-zero reference entries, e.g. PAF maps in pose estimation)
        'rel_norm': (diff.mean() / ref_abs).item(),
        'time_online': time_online,
    }


def _connect(port, host='127.0.0.1', timeout=60.0):
    """Connect to the server, retrying until it starts listening."""
    deadline = time.time() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=1.0)
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)


def _read(f):
    f.seek(0)
    return f.read().decode('utf-8', 'ignore')


def _free_port():
    s = socket.create_server(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    time.sleep(0.05)  # give the OS a moment to release the ephemeral port
    return port


def show_report(r):
    print("[{}] model {}: abs diff mean {:.3e} / max {:.3e}, rel(diff) {:.3e}, rel(norm) {:.3e}, online {:.3f}s".format(
        r['protocol'], r['model'], r['abs_diff_mean'], r['abs_diff_max'],
        r['rel_diff_mean'], r['rel_norm'], r['time_online']))
    return r


def check_threshold(r, protocol_rtol=1e-5, r_rtol=5e-2):
    """plaintext must match the local result exactly (rtol on the result scale);
    masked protocols allow a loose bound (FP error accumulates over layers)."""
    if r['protocol'] == 'plaintext':
        assert r['rel_norm'] < protocol_rtol, (
            "plaintext system result deviates from local result [{}]:\nresult {}\nreference {}".format(
                r['model'], r['result'], r['reference']))
    else:
        assert r['rel_norm'] < r_rtol, (
            "system result deviates from local result too much [{} {}]:\nresult {}\nreference {}".format(
                r['protocol'], r['model'], r['result'], r['reference']))
    return r
