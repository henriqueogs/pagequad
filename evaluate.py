"""Held-out evaluation on fresh synthetic scenes (seeds the training never used).

    python evaluate.py [--weights net.pt] [--n 1000] [--json results/metrics.json]

Reports mask IoU, and how often the 4 corners recovered from the mask land within 2% / 4% / 8% of the frame diagonal of the truth,
overall and split by whether the whole sheet is inside the photo (a sheet cut by the frame border has corners that no mask can show).
"""
import argparse, json, math, os
import numpy as np
import torch
from gen import sample, S
from model import Net
from train import quad_from_mask, order


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--weights', default='net.pt'); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--json', default='results/metrics.json')
    ap.add_argument('--seed', type=int, default=7_000_000)
    a = ap.parse_args()
    net = Net(); net.load_state_dict(torch.load(a.weights)); net.eval()
    rows = []
    with torch.no_grad():
        for i in range(a.n):
            x, m, q = sample(a.seed + i)
            p = torch.sigmoid(net(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255))[0, 0].numpy()
            pr, gt = p > .5, m > .5; inter = (pr & gt).sum(); uni = (pr | gt).sum(); iou = float(inter / uni) if uni else 1.0
            r = dict(iou=iou, has=q is not None)
            if q is not None:
                r['inframe'] = bool(((q >= -.002) & (q <= 1.002)).all())
                qq = quad_from_mask(p); r['err'] = None if qq is None else float(np.linalg.norm(qq - order(q), axis=1).mean() / math.sqrt(2))
            rows.append(r)

    def rate(sel, t):
        s = [r for r in sel if r['has']]
        return round(sum(1 for r in s if r['err'] is not None and r['err'] < t) / max(1, len(s)), 4), len(s)

    pages = [r for r in rows if r['has']]; inf = [r for r in pages if r['inframe']]; outf = [r for r in pages if not r['inframe']]
    errs = sorted(r['err'] for r in inf if r['err'] is not None)
    res = dict(n=a.n, mask_iou_mean=round(float(np.mean([r['iou'] for r in rows])), 4),
               corners_within={f'{int(t*100)}pct': {'all_pages': rate(pages, t)[0], 'whole_sheet_in_photo': rate(inf, t)[0], 'sheet_cut_by_border': rate(outf, t)[0]} for t in (.02, .04, .08)},
               counts=dict(pages=len(pages), whole_sheet_in_photo=len(inf), sheet_cut_by_border=len(outf), no_page=a.n - len(pages)),
               median_corner_error_pct_inframe=round(100 * float(np.median(errs)), 2) if errs else None,
               no_quad_found_inframe=sum(1 for r in inf if r['err'] is None))
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True); json.dump(res, open(a.json, 'w'), indent=2)


if __name__ == '__main__':
    main()
