"""Train the page-segmentation net on synthetic scenes (CPU).
Data comes from a few light worker processes (numpy + cv2 only; torch is imported inside main() so the workers stay small)."""
import argparse, math, time
from collections import deque
import numpy as np
import cv2
from gen import sample, S


def make_batch(args):
    base, bs = args
    xs, ms, qs, hs = [], [], [], []
    for i in range(bs):
        x, m, q = sample(base + i)
        xs.append(x); ms.append(m); qs.append(q if q is not None else np.zeros((4, 2), np.float32)); hs.append(q is not None)
    return np.stack(xs), np.stack(ms), np.stack(qs), np.array(hs)


def order(q):
    s, d = q.sum(1), q[:, 0] - q[:, 1]
    return np.array([q[s.argmin()], q[d.argmax()], q[s.argmax()], q[d.argmin()]])


def quad_from_mask(m):
    """mask (S,S) -> 4 ordered corners in [0,1] or None: contour -> convex hull -> approxPolyDP sweep, else minAreaRect (same idea as the app)."""
    b = (m > .5).astype(np.uint8)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < .08 * S * S: return None
    h = cv2.convexHull(c); per = cv2.arcLength(h, True); pts = None
    for eps in (.012, .02, .03, .045, .065, .09):
        ap = cv2.approxPolyDP(h, eps * per, True)
        if len(ap) == 4: pts = ap[:, 0, :].astype(np.float32); break
        if len(ap) < 4: break
    if pts is None: pts = cv2.boxPoints(cv2.minAreaRect(h)).astype(np.float32)
    return order(pts) / S


def main():
    import torch, torch.nn.functional as F
    from multiprocessing import get_context
    from model import Net
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--workers', type=int, default=5); ap.add_argument('--out', default='net.pt')
    ap.add_argument('--resume', default=None)
    a = ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0)
    net = Net()
    if a.resume: net.load_state_dict(torch.load(a.resume))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=.1)

    @torch.no_grad()
    def evaluate(batches):
        net.eval(); ious, ok, tot = [], 0, 0
        for x, m, q, has in batches:
            p = torch.sigmoid(net(torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255)).numpy()[:, 0]
            for i in range(len(x)):
                pr = p[i] > .5; gt = m[i] > .5; inter = (pr & gt).sum(); uni = (pr | gt).sum(); ious.append(inter / uni if uni else 1.0)
                if has[i]:
                    tot += 1; qq = quad_from_mask(p[i])
                    if qq is not None and np.linalg.norm(qq - order(q[i]), axis=1).mean() / math.sqrt(2) < .04: ok += 1
        net.train(); return float(np.mean(ious)), ok / max(1, tot)

    pool = get_context('spawn').Pool(a.workers)
    val = [make_batch((5_000_000 + k * 100, 100)) for k in range(4)]  # 400 fixed held-out scenes
    window = deque(pool.apply_async(make_batch, ((i * a.bs, a.bs),)) for i in range(a.workers * 2)); nxt = a.workers * 2
    t0 = time.time(); run = 0.0
    for step in range(1, a.steps + 1):
        x, m, q, has = window.popleft().get()
        if nxt < a.steps: window.append(pool.apply_async(make_batch, ((nxt * a.bs, a.bs),))); nxt += 1
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255; m = torch.from_numpy(m)[:, None]
        logit = net(x)
        bce = F.binary_cross_entropy_with_logits(logit, m)
        p = torch.sigmoid(logit); inter = (p * m).sum((1, 2, 3)); dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
        loss = bce + dice.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = loss.item() if step == 1 else .98 * run + .02 * loss.item()
        if step % 100 == 0:
            el = time.time() - t0; print(f'step {step}/{a.steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.5f} {el/step:.3f}s/step eta {(a.steps-step)*el/step/60:.1f}min', flush=True)
        if step % 1000 == 0 or step == a.steps:
            iou, ok = evaluate(val); print(f'  VAL step {step}: IoU {iou:.4f}  quad<4%: {ok:.3f}', flush=True); torch.save(net.state_dict(), a.out)
    pool.terminate(); print('done', flush=True)


if __name__ == '__main__':
    main()
