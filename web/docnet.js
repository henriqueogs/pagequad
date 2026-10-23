// Inference engine for our own page-segmentation net (docmodel/). Plain JS, no dependencies:
// conv (full / depthwise / pointwise) with folded BatchNorm + ReLU6, nearest x2 upsampling, concat, global-context add,
// bilinear x2 and sigmoid. Tensors are Float32Array in CHW order. The layer list and weights come from export.py.
const DocNet = (() => {
  let spec = null, W = null;
  const clamp6 = v => v < 0 ? 0 : v > 6 ? 6 : v;
  const T = (c, h, w, d) => ({c, h, w, d: d || new Float32Array(c * h * w)});

  function conv(x, o) {
    const p = o.k >> 1, ho = Math.floor((x.h + 2 * p - o.k) / o.s) + 1, wo = Math.floor((x.w + 2 * p - o.k) / o.s) + 1, y = T(o.co, ho, wo), N = ho * wo;
    const w = W.subarray(o.w), b = W.subarray(o.b);
    if (o.k === 1) { // pointwise: for each output channel accumulate scaled input planes (contiguous inner loop)
      for (let co = 0; co < o.co; co++) { const yo = y.d.subarray(co * N, co * N + N); yo.fill(b[co]);
        for (let ci = 0; ci < o.ci; ci++) { const wv = w[co * o.ci + ci]; if (wv === 0) continue; const xi = x.d.subarray(ci * N, ci * N + N); for (let i = 0; i < N; i++) yo[i] += wv * xi[i]; } }
    } else if (o.g > 1) { // depthwise 3x3
      for (let c = 0; c < o.co; c++) { const xc = c * x.h * x.w, wc = c * 9, bc = b[c];
        for (let oy = 0; oy < ho; oy++) for (let ox = 0; ox < wo; ox++) { let s = bc;
          for (let ky = 0; ky < 3; ky++) { const iy = oy * o.s - p + ky; if (iy < 0 || iy >= x.h) continue;
            for (let kx = 0; kx < 3; kx++) { const ix = ox * o.s - p + kx; if (ix < 0 || ix >= x.w) continue; s += w[wc + ky * 3 + kx] * x.d[xc + iy * x.w + ix]; } }
          y.d[c * N + oy * wo + ox] = s; } }
    } else { // full kxk
      for (let co = 0; co < o.co; co++) for (let oy = 0; oy < ho; oy++) for (let ox = 0; ox < wo; ox++) { let s = b[co];
        for (let ci = 0; ci < o.ci; ci++) { const xc = ci * x.h * x.w, wc = (co * o.ci + ci) * o.k * o.k;
          for (let ky = 0; ky < o.k; ky++) { const iy = oy * o.s - p + ky; if (iy < 0 || iy >= x.h) continue;
            for (let kx = 0; kx < o.k; kx++) { const ix = ox * o.s - p + kx; if (ix < 0 || ix >= x.w) continue; s += w[wc + ky * o.k + kx] * x.d[xc + iy * x.w + ix]; } } }
        y.d[co * N + oy * wo + ox] = s; }
    }
    if (o.act) for (let i = 0; i < y.d.length; i++) y.d[i] = clamp6(y.d[i]);
    return y;
  }
  const up = x => { const y = T(x.c, x.h * 2, x.w * 2); for (let c = 0; c < x.c; c++) for (let yy = 0; yy < y.h; yy++) { const sy = yy >> 1; for (let xx = 0; xx < y.w; xx++) y.d[(c * y.h + yy) * y.w + xx] = x.d[(c * x.h + sy) * x.w + (xx >> 1)]; } return y; };
  const upBilinear = x => { const y = T(x.c, x.h * 2, x.w * 2); // align_corners=false, scale 2
    for (let c = 0; c < x.c; c++) for (let yy = 0; yy < y.h; yy++) { let sy = (yy + .5) / 2 - .5; sy = Math.max(0, sy); const y0 = Math.min(x.h - 1, Math.floor(sy)), y1 = Math.min(x.h - 1, y0 + 1), fy = sy - y0;
      for (let xx = 0; xx < y.w; xx++) { let sx = (xx + .5) / 2 - .5; sx = Math.max(0, sx); const x0 = Math.min(x.w - 1, Math.floor(sx)), x1 = Math.min(x.w - 1, x0 + 1), fx = sx - x0, b = c * x.h * x.w;
        y.d[(c * y.h + yy) * y.w + xx] = (x.d[b + y0 * x.w + x0] * (1 - fx) + x.d[b + y0 * x.w + x1] * fx) * (1 - fy) + (x.d[b + y1 * x.w + x0] * (1 - fx) + x.d[b + y1 * x.w + x1] * fx) * fy; } }
    return y; };
  const cat = (a, b) => { const y = T(a.c + b.c, a.h, a.w); y.d.set(a.d); y.d.set(b.d, a.d.length); return y; };
  const gap = x => { const y = T(x.c, 1, 1), N = x.h * x.w; for (let c = 0; c < x.c; c++) { let s = 0; for (let i = 0; i < N; i++) s += x.d[c * N + i]; y.d[c] = s / N; } return y; };
  const addbc = (x, g) => { const y = T(x.c, x.h, x.w), N = x.h * x.w; for (let c = 0; c < x.c; c++) for (let i = 0; i < N; i++) y.d[c * N + i] = x.d[c * N + i] + g.d[c]; return y; };

  return {
    async load(base) { spec = await (await fetch(base + '.json')).json(); W = new Float32Array(await (await fetch(base + '.bin')).arrayBuffer()); return this; },
    get ready() { return !!spec; }, get size() { return spec.size; },
    // rgb: Uint8 RGB(A) pixels of a size x size image; returns the sheet probability map (size x size Float32Array)
    run(pix, channels) {
      const n = spec.size, x = T(3, n, n), N = n * n;
      for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) x.d[c * N + i] = pix[i * channels + c] / 255;
      const t = {x};
      for (const o of spec.ops) {
        if (o.op === 'affine') t[o.o] = (() => { const y = T(x.c, x.h, x.w); for (let i = 0; i < x.d.length; i++) y.d[i] = x.d[i] * o.mul + o.add; return y; })();
        else if (o.op === 'conv') t[o.o] = conv(t[o.i], o);
        else if (o.op === 'up') t[o.o] = up(t[o.i]);
        else if (o.op === 'upbilinear') t[o.o] = upBilinear(t[o.i]);
        else if (o.op === 'cat') t[o.o] = cat(t[o.ins[0]], t[o.ins[1]]);
        else if (o.op === 'gap') t[o.o] = gap(t[o.i]);
        else if (o.op === 'addbc') t[o.o] = addbc(t[o.i], t[o.a]);
        else if (o.op === 'sigmoid') { const s = t[o.i], y = T(s.c, s.h, s.w); for (let i = 0; i < s.d.length; i++) y.d[i] = 1 / (1 + Math.exp(-s.d[i])); t[o.o] = y; }
      }
      return t.prob.d;
    },
  };
})();
