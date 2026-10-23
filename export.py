"""Export the trained net for the app: BatchNorm folded into the convs, float32 weights in one .bin, the layer list in a .json,
plus a test vector (input image + the PyTorch output) so the JavaScript engine can be checked number by number."""
import json, sys
import numpy as np
import torch
from model import Net
from gen import sample

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'net.pt'
out = sys.argv[2] if len(sys.argv) > 2 else 'web/docnet'
net = Net(); net.load_state_dict(torch.load(ckpt)); net.eval()

ops, chunks, off = [], [], 0


def push(arr):
    global off
    a = np.ascontiguousarray(arr, dtype=np.float32).ravel(); chunks.append(a); o = off; off += a.size; return o


def fold(seq):
    conv, bn = seq[0], seq[1]
    w = conv.weight.detach().numpy(); g = bn.weight.detach().numpy(); b = bn.bias.detach().numpy(); m = bn.running_mean.numpy(); v = bn.running_var.numpy()
    s = g / np.sqrt(v + bn.eps); return w * s[:, None, None, None], b - m * s


def conv_op(src, dst, seq, act=1):
    w, b = fold(seq); conv = seq[0]; co, cig, k, _ = w.shape
    ops.append(dict(op='conv', i=src, o=dst, k=k, s=conv.stride[0], g=conv.groups, ci=cig * conv.groups, co=co, w=push(w), b=push(b), act=act))


def ds_op(src, dst, m):
    conv_op(src, dst + '_dw', m.dw); conv_op(dst + '_dw', dst, m.pw)


ops.append(dict(op='affine', i='x', o='x', mul=2.0, add=-1.0))
conv_op('x', 'a', net.s0)
ds_op('a', 'b', net.s1)
ds_op('b', 'c1', net.s2[0]); ds_op('c1', 'c', net.s2[1])
ds_op('c', 'd1', net.s3[0]); ds_op('d1', 'd', net.s3[1])
ds_op('d', 'e1', net.s4[0]); ds_op('e1', 'e0', net.s4[1])
ops.append(dict(op='gap', i='e0', o='g0')); conv_op('g0', 'g', net.ctx); ops.append(dict(op='addbc', i='e0', a='g', o='e'))
ops.append(dict(op='up', i='e', o='u3')); ops.append(dict(op='cat', ins=['u3', 'd'], o='k3')); ds_op('k3', 'm3', net.d3)
ops.append(dict(op='up', i='m3', o='u2')); ops.append(dict(op='cat', ins=['u2', 'c'], o='k2')); ds_op('k2', 'm2', net.d2)
ops.append(dict(op='up', i='m2', o='u1')); ops.append(dict(op='cat', ins=['u1', 'b'], o='k1')); ds_op('k1', 'm1', net.d1)
hw = net.head.weight.detach().numpy(); hb = net.head.bias.detach().numpy()
ops.append(dict(op='conv', i='m1', o='h', k=1, s=1, g=1, ci=24, co=1, w=push(hw), b=push(hb), act=0))
ops.append(dict(op='upbilinear', i='h', o='logit')); ops.append(dict(op='sigmoid', i='logit', o='prob'))

spec = dict(size=128, ops=ops, params=off)
json.dump(spec, open(out + '.json', 'w')); np.concatenate(chunks).tofile(out + '.bin')
print('ops', len(ops), 'floats', off, 'bytes', off * 4)

# test vectors: 3 scenes (uint8 RGB 128x128 + torch probability map) for the JS engine check
tv = []
for i in range(3):
    x, m, q = sample(9_000_000 + i)
    with torch.no_grad(): p = torch.sigmoid(net(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255))[0, 0].numpy()
    tv.append(dict(x=x.ravel().tolist(), p=np.round(p, 5).ravel().tolist()))
json.dump(tv, open(out + '.test.json', 'w'))
print('test vectors written')
