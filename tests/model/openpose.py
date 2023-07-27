from model import openpose
import torch
import sys
import time


def test_body_model(n, weight_path, device):
    mo = openpose.op_impl.bodypose_model()
    if weight_path is not None:
        openpose.load_model(mo, torch.load(weight_path))
    mo = mo.to(device)
    mo.eval()

    ms = openpose.build('body', weight_path)
    ms = ms.to(device)
    ms.eval()

    x = torch.randn(n, 3, 368, 368, device=device)
    with torch.no_grad():
        t0 = time.time()
        yo = mo(x)
        t1 = time.time()
        ys = ms(x)
        t2 = time.time()
    heatmap_o, paf_o = yo
    # heatmap_s, paf_s = ys[:, :38], ys[:, 38:]
    heatmap_s, paf_s = openpose.sep_body_result(ys)

    print('time: mo {:.3f}s, ms {:.3f}s'.format(t1 - t0, t2 - t1))
    print(heatmap_s.shape, heatmap_o.shape)
    print(paf_s.shape, paf_o.shape)
    diff_h = (heatmap_s - heatmap_o).cpu()
    diff_p = (paf_s - paf_o).cpu()
    print(diff_p.max(), diff_p.min(), diff_p.mean(), diff_p.square().mean().sqrt())
    print(diff_h.max(), diff_h.min(), diff_h.mean(), diff_h.square().mean().sqrt())


def test_hand_model(n, weight_path, device):
    mo = openpose.op_impl.handpose_model()
    if weight_path is not None:
        openpose.load_model(mo, torch.load(weight_path))
    mo = mo.to(device)
    mo.eval()

    ms = openpose.build('hand', weight_path)
    ms = ms.to(device)
    ms.eval()

    x = torch.randn(n, 3, 368, 368, device=device)
    with torch.no_grad():
        t0 = time.time()
        yo = mo(x)
        t1 = time.time()
        ys = ms(x)
        t2 = time.time()
    
    print('time: mo {:.3f}s, ms {:.3f}s'.format(t1 - t0, t2 - t1))
    print(ys.shape, yo.shape)
    diff = (ys - yo).cpu()
    print(diff.max(), diff.min(), diff.mean(), diff.square().mean().sqrt())


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: python openpose.py [n=1] [device=cpu] [weight_file_prefix]')
        sys.exit(1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    prefix = sys.argv[3] if len(sys.argv) > 3 else None
    assert device in ['cpu', 'cuda']

    wbody = '{}/body_pose_model.pth'.format(prefix) if prefix is not None else None
    whand = '{}/hand_pose_model.pth'.format(prefix) if prefix is not None else None
    print('test body model')
    test_body_model(n, wbody, device)
    print('test hand model')
    test_hand_model(n, whand, device)
