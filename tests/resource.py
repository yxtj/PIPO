# -*- coding: utf-8 -*-
"""Resource round trip: generate (mode=offline) -> restore (mode=online)
-> compare against the plaintext local result.

Usage: python tests/resource.py [protocol] [model_name]
       (defaults: protocol padscaleshuffle, model poc-1)

For the given protocol/model this script
1. starts `python -m src.main server --mode offline` as a subprocess and runs
   the client in this process, so both roles interactively generate and
   persist their own half of one resource set,
2. starts `python -m src.main server --mode online` as a subprocess that
   loads that set,
3. runs the client in this process with the restored client resources and
   performs two online rounds on the same input,
4. checks that the two rounds are bitwise identical (same resource + same
   input => identical intermediate results) and that the result matches the
   local plaintext reference within the protocol's error bound.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models import registry
from session import _connect, _free_port, _read


def _wait_server(server, err):
    server.wait(timeout=300)
    server_err = _read(err)
    if server.returncode not in (0, None):
        raise RuntimeError('server run failed (code {}):\n{}'.format(
            server.returncode, server_err))
    return server_err


def run_roundtrip(model_name, protocol='padscaleshuffle', seed=0, device='cpu'):
    from src.common.config import load_config, set_config

    # the config must be committed before the protocol modules are
    # dynamically imported (src.protocol)
    res_dir = tempfile.mkdtemp(prefix='pipo_res_')
    cfg = load_config()
    cfg.server.protocol.name = protocol
    cfg.common.use_he = False
    cfg.common.device = device
    cfg.common.resource.dir = res_dir
    cfg.common.resource.index = 0
    set_config(cfg)

    from src.system.resource import list_indices, load_resource_set_client

    torch.manual_seed(seed)
    inshape, model = registry.load(model_name, seed=seed, device=device)

    try:
        # 1) interactive offline generation: the server subprocess persists
        #    its half, the in-process client persists its own half
        port = _free_port()
        out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        args = [sys.executable, '-m', 'src.main', 'server', '--mode', 'offline',
                '--res-dir', res_dir, '--model', model_name, '--protocol', protocol,
                '--port', str(port), '--device', device]
        server = subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdout=out, stderr=err)
        try:
            from src.system.runner import run_client

            run_client('127.0.0.1', port, None, cfg,
                       do_offline=True, do_online=False)
        finally:
            _wait_server(server, err)
            out.close()
            err.close()

        indices = list_indices(res_dir, model_name)
        assert indices == [1], 'expected exactly one generated resource set, got {}'.format(indices)

        # both files must carry identical identity fields
        server_payload = torch.load(Path(res_dir) / model_name / 'res_0001_server.pt',
                                    weights_only=True)
        client_payload = torch.load(Path(res_dir) / model_name / 'res_0001_client.pt',
                                    weights_only=True)
        for key in ('version', 'model_name', 'model_info_hash', 'protocol', 'protocol_args'):
            assert server_payload[key] == client_payload[key], \
                'identity field {!r} differs between the server and client files'.format(key)
        server_entries = server_payload['layers']
        client_entries = client_payload['layers']
        assert len(server_entries) == len(client_entries) > 0, 'layer entry count mismatch'

        # 2) server subprocess in online mode, restoring the generated set
        port = _free_port()
        data = torch.rand(1, *inshape)
        if device.startswith('cuda') and torch.cuda.is_available():
            data = data.to(device)

        out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        args = [sys.executable, '-m', 'src.main', 'server', '--mode', 'online',
                '--res-dir', res_dir, '--res-index', '1',
                '--model', model_name, '--n', '2', '--protocol', protocol,
                '--port', str(port), '--device', device]
        server = subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdout=out, stderr=err)
        try:
            from src.system import Client

            c = _connect(port)
            client = Client(c, he=None)
            client.setup()
            assert client.resource == 1, \
                'server announced resource set {}'.format(client.resource)
            # 3) restore the client side from the announced set
            entries = load_resource_set_client(res_dir, client.model_name,
                                               client.resource, client.info)
            client.setup_layers(res=entries)

            # two online rounds on the same input must be identical
            with torch.no_grad():
                res1 = client.online(data)
                res2 = client.online(data)
            assert torch.equal(res1, res2), \
                'same resource + same input produced different results (max diff {})'.format(
                    (res1 - res2).abs().max())
        finally:
            _wait_server(server, err)
            out.close()
            err.close()

        # 4) plaintext reference
        with torch.no_grad():
            ref = model(data)
        diff = (res1 - ref).abs()
        rel = diff / ref.abs().clamp_min(1e-12)
        ref_abs = ref.abs().mean().clamp_min(1e-12).item()
        return {
            'model': model_name,
            'protocol': protocol,
            'abs_diff_mean': diff.mean().item(),
            'abs_diff_max': diff.max().item(),
            'rel_norm': (diff.mean() / ref_abs).item(),
        }
    finally:
        shutil.rmtree(res_dir, ignore_errors=True)


def check_threshold(r, protocol_rtol=1e-5, r_rtol=5e-2):
    """plaintext must match the local result exactly; masked protocols allow
    the same loose FP bound as session.check_threshold."""
    if r['protocol'] == 'plaintext':
        assert r['rel_norm'] < protocol_rtol, (
            'plaintext result deviates from local result:\n{}'.format(r))
    else:
        assert r['rel_norm'] < r_rtol, (
            'restored-resource result deviates from local result too much:\n{}'.format(r))
    return r


def main():
    argv = [a for a in sys.argv[1:] if not str(a).startswith('-')]
    protocol = argv[0] if len(argv) >= 1 else 'padscaleshuffle'
    model_name = argv[1] if len(argv) >= 2 else 'poc-1'
    if not model_name.startswith('poc-'):
        model_name = 'poc-' + model_name

    print('resource round trip: model {} / protocol {}'.format(model_name, protocol))
    r = check_threshold(run_roundtrip(model_name, protocol))
    print('[{}] model {}: abs diff mean {:.3e} / max {:.3e}, rel(norm) {:.3e} '
          '(rounds identical, resource restored)'.format(
              r['protocol'], r['model'], r['abs_diff_mean'], r['abs_diff_max'], r['rel_norm']))
    print('OK')


if __name__ == '__main__':
    main()