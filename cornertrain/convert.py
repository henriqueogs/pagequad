"""ONNX (KennethTM/document_corner_detector, MIT) -> PyTorch module, weights untouched. Clip nodes with an empty min/max input are patched for onnx2torch."""
import sys, onnx, numpy as np, torch
from onnx import numpy_helper
from onnx2torch import convert
om = onnx.load(sys.argv[1]); g = om.graph
g.initializer.extend([numpy_helper.from_array(np.array(-3.4e38, 'float32'), '__neginf'), numpy_helper.from_array(np.array(3.4e38, 'float32'), '__posinf')])
for n in g.node:
    if n.op_type == 'Clip':
        ins = list(n.input)
        while len(ins) < 3: ins.append('')
        if ins[1] == '': ins[1] = '__neginf'
        if ins[2] == '': ins[2] = '__posinf'
        del n.input[:]; n.input.extend(ins)
m = convert(om); m.eval(); torch.save(m, sys.argv[2]); print('saved', sys.argv[2])
