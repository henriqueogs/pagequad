"""Fine-tune the page-corner heatmap network (MobileNetV3-small U-Net, 4 heatmaps: TL TR BR BL) on real document photos."""
import sys, time, math, random, numpy as np, cv2, torch
N = 224
def order(p):
    s = p.sum(1); d = p[:, 0] - p[:, 1]
    return np.array([p[s.argmin()], p[d.argmax()], p[s.argmax()], p[d.argmin()]], np.float32)
def letterbox(im, c):
    h, w = im.shape[:2]; k = N / max(h, w); nh, nw = max(1, int(h * k)), max(1, int(w * k))
    sm = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    y0, x0 = -((nh - N) // 2) if nh < N else 0, -((nw - N) // 2) if nw < N else 0
    pad = np.zeros((N, N, 3), np.uint8); pad[y0:y0 + nh, x0:x0 + nw] = sm
    return pad, (None if c is None else c * k + np.array([x0, y0], np.float32))
def heat(c, sig=3.0):
    hm = np.zeros((4, N, N), np.float32)
    if c is None: return hm
    ys, xs = np.mgrid[0:N, 0:N]
    for i, (x, y) in enumerate(c): hm[i] = np.exp(-((xs + .5 - x) ** 2 + (ys + .5 - y) ** 2) / (2 * sig * sig))
    return hm
def augment(im, c, rng):
    h, w = im.shape[:2]
    if rng.random() < .35: k = rng.integers(1, 4); im = np.ascontiguousarray(np.rot90(im, k)); 
    else: k = 0
    if k:  # rotate corners the same way (np.rot90 counter-clockwise, k times)
        for _ in range(k):
            if c is not None: c = np.stack([c[:, 1], w - c[:, 0]], 1)
            h, w = w, h
    if rng.random() < .5:
        im = im[:, ::-1]; 
        if c is not None: c = np.stack([w - c[:, 0], c[:, 1]], 1)
    # small rotation + scale about the centre
    ang = rng.normal(0, 8) if rng.random() < .6 else 0; sc = float(np.exp(rng.normal(0, .12)))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc); bg = int(rng.integers(0, 255)) if rng.random() < .5 else 0
    im = cv2.warpAffine(np.ascontiguousarray(im), M, (w, h), borderValue=(bg, bg, bg))
    if c is not None: c = (np.hstack([c, np.ones((4, 1), np.float32)]) @ M.T).astype(np.float32)
    # random crop around the page (page small or filling the frame) -- or keep whole frame
    if c is not None and rng.random() < .7:
        x0, y0, x1, y1 = c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max(); bw, bh = x1 - x0, y1 - y0
        mx, my = bw * rng.uniform(-.0, 1.2), bh * rng.uniform(-.0, 1.2)
        cx0, cy0 = int(max(0, x0 - mx * rng.random())), int(max(0, y0 - my * rng.random())); cx1, cy1 = int(min(w, x1 + mx * rng.random())), int(min(h, y1 + my * rng.random()))
        if cx1 - cx0 > 40 and cy1 - cy0 > 40 and cx0 <= x0 + 2 and cy0 <= y0 + 2 and cx1 >= x1 - 2 and cy1 >= y1 - 2:
            im = im[cy0:cy1, cx0:cx1]; c = c - np.array([cx0, cy0], np.float32)
    im = np.ascontiguousarray(im).astype(np.float32)
    # photometric
    im = im * rng.uniform(.6, 1.3) + rng.uniform(-30, 30); im = np.clip(im, 0, 255)
    if rng.random() < .4: g = rng.uniform(.6, 1.6); im = 255 * (im / 255) ** g
    if rng.random() < .3: im = cv2.GaussianBlur(im, (0, 0), rng.uniform(.5, 2))
    if rng.random() < .3: im = im + rng.normal(0, rng.uniform(2, 10), im.shape)
    return np.clip(im, 0, 255).astype(np.uint8), (None if c is None else order(c))
class DS(torch.utils.data.Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        b, c, _ = self.items[i]; im = cv2.imdecode(np.frombuffer(b, np.uint8), 1); rng = np.random.default_rng((i * 7919 + int(time.time() * 1000)) % 2 ** 32)
        if self.train: im, c = augment(im, None if c is None else c.copy(), rng)
        pad, c2 = letterbox(im, c)
        x = ((cv2.cvtColor(pad, cv2.COLOR_BGR2RGB) / 255.0 - [.485, .456, .406]) / [.229, .224, .225]).astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(x), torch.from_numpy(heat(c2)), (torch.from_numpy(c2) if c2 is not None else torch.full((4, 2), -1.0)), torch.tensor(float(c is not None))
def evaluate(model, loader):
    model.eval(); errs = []
    with torch.no_grad():
        for x, _, c, has in loader:
            hm = model(x); 
            for b in range(len(x)):
                if not has[b]: continue
                pr = []
                for k in range(4):
                    i = int(hm[b, k].argmax()); pr.append([(i % N) + .5, (i // N) + .5])
                errs.append(float(np.linalg.norm(np.array(pr) - c[b].numpy(), axis=1).max() / math.hypot(N, N)))   # worst corner / frame diagonal (letterboxed square)
    e = np.array(errs); model.train()
    return float(np.median(e)), float((e < .03).mean()), float((e > .10).mean())
if __name__ == '__main__':
    train_npy, val_npy, epochs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.set_num_threads(12); random.seed(0); torch.manual_seed(0)
    tr = list(np.load(train_npy, allow_pickle=True)); va = list(np.load(val_npy, allow_pickle=True))
    print(len(tr), 'train', len(va), 'val', flush=True)
    model = torch.load(sys.argv[5] if len(sys.argv) > 5 else 'start.pt', weights_only=False); model.train()
    tl = torch.utils.data.DataLoader(DS(tr, True), batch_size=16, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(DS(va, False), batch_size=16, num_workers=0)
    print('start', evaluate(model, vl), flush=True)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4); total = epochs * len(tl); step = 0; best = 9
    for ep in range(epochs):
        t0 = time.time(); run = 0
        for x, hm, _, _ in tl:
            lr = 2e-4 * (.5 * (1 + math.cos(math.pi * step / total))) * min(1, (step + 1) / 50); [g.__setitem__('lr', lr) for g in opt.param_groups]
            p = model(x); loss = ((p - hm) ** 2 * (1 + 30 * hm)).mean() * 100
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; run += float(loss)
            if step % 20 == 0: print(f'ep{ep} step{step}/{total} loss {run / 20:.3f} {time.time() - t0:.0f}s', flush=True); run = 0
        med, ok, bad = evaluate(model, vl); print(f'== epoch {ep}: median {med:.4f}  <3% {ok:.2%}  >10% {bad:.2%}  ({time.time() - t0:.0f}s)', flush=True)
        score = med + bad
        if score < best: best = score; torch.save(model, out); print('saved', out, flush=True)
