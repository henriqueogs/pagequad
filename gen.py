"""Synthetic training scenes for the page-segmentation model.

Every sample is a photo-like frame with one sheet of paper on some surface, plus the usual trouble:
light/dark/patterned surfaces, soft shadows, fingers and objects over the page edge, uneven light,
camera blur, sensor noise and JPEG artefacts. The label is the mask of the whole sheet (also under fingers).
Frames are rendered with random aspect ratios and then squashed to SxS, the same way the app feeds the network.
"""
import math
import numpy as np
import cv2

S = 128  # network input / output resolution


def rand_color(rng, mode):
    if mode == 'light':
        base = rng.uniform(205, 245); tint = rng.uniform(-14, 14, 3)
        return np.clip(base + tint, 0, 255)
    if mode == 'dark':
        return np.clip(rng.uniform(15, 75) + rng.uniform(-10, 10, 3), 0, 255)
    if mode == 'mid':
        return np.clip(rng.uniform(80, 190) + rng.uniform(-25, 25, 3), 0, 255)
    return rng.uniform(40, 230, 3)  # saturated / any


def smooth_noise(h, w, rng, octaves=4):
    out = np.zeros((h, w), np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = int(rng.integers(3, 8)) * (2 ** o)
        out += amp * cv2.resize(rng.random((s, s), dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= .5
    return out / tot


def background(h, w, rng):
    mode = rng.choice(['light', 'light', 'light', 'dark', 'mid', 'sat'], p=[.28, .17, .1, .17, .18, .1])
    c1, c2 = rand_color(rng, mode), rand_color(rng, mode)
    kind = rng.choice(['solid', 'noise', 'wood', 'cloth', 'checker', 'stripes', 'dots', 'gradient'], p=[.16, .2, .14, .1, .1, .1, .06, .14])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == 'solid':
        img = np.ones((h, w, 3), np.float32) * c1
    elif kind == 'gradient':
        a = rng.uniform(0, 2 * math.pi); t = (np.cos(a) * xx / w + np.sin(a) * yy / h); t = (t - t.min()) / (np.ptp(t) + 1e-6)
        img = c1 * (1 - t[..., None]) + c2 * t[..., None]
    elif kind == 'noise':
        t = smooth_noise(h, w, rng, 5)[..., None]; img = c1 * (1 - t) + c2 * t
    elif kind == 'wood':
        ang = rng.uniform(0, math.pi); u = np.cos(ang) * xx + np.sin(ang) * yy; v = -np.sin(ang) * xx + np.cos(ang) * yy
        t = .5 + .5 * np.sin(v / rng.uniform(2.5, 9) + 6 * smooth_noise(h, w, rng, 3)) * .6 + .25 * smooth_noise(h, w, rng, 5)
        t = np.clip(t, 0, 1)[..., None]; img = c1 * (1 - t) + c2 * t
    elif kind == 'cloth':
        t = (.5 + .5 * np.sin(xx / rng.uniform(1.2, 3))) * (.5 + .5 * np.sin(yy / rng.uniform(1.2, 3)))
        t = (.65 * t + .35 * smooth_noise(h, w, rng, 4))[..., None]; img = c1 * (1 - t) + c2 * t
    elif kind == 'checker':
        s = rng.uniform(10, 38); ang = rng.uniform(0, math.pi / 2); u = (np.cos(ang) * xx + np.sin(ang) * yy) / s; v = (-np.sin(ang) * xx + np.cos(ang) * yy) / s
        t = ((np.floor(u) + np.floor(v)) % 2)[..., None]; img = c1 * (1 - t) + c2 * t
    elif kind == 'stripes':
        s = rng.uniform(6, 30); ang = rng.uniform(0, math.pi); u = (np.cos(ang) * xx + np.sin(ang) * yy) / s
        t = (np.floor(u) % 2)[..., None]; img = c1 * (1 - t) + c2 * t
    else:  # dots
        s = rng.uniform(9, 24); t = (((xx % s) - s / 2) ** 2 + ((yy % s) - s / 2) ** 2 < (s * .28) ** 2).astype(np.float32)[..., None]; img = c1 * (1 - t) + c2 * t
    img = img + (rng.random((h, w, 1), dtype=np.float32) - .5) * rng.uniform(0, 10)  # grain
    return np.clip(img, 0, 255)


def paper_sheet(pw, ph, rng):
    mode = rng.choice(['white', 'cream', 'gray', 'color'], p=[.55, .22, .13, .10])
    if mode == 'white': col = rng.uniform(238, 255) + rng.uniform(-4, 3, 3)
    elif mode == 'cream': col = np.array([rng.uniform(240, 252), rng.uniform(232, 246), rng.uniform(205, 232)])
    elif mode == 'gray': col = rng.uniform(190, 232) + rng.uniform(-5, 5, 3)
    else: col = rng.uniform(150, 245, 3)
    img = np.ones((ph, pw, 3), np.float32) * col
    ink = rng.uniform(15, 70)
    mx, my = int(pw * rng.uniform(.06, .14)), int(ph * rng.uniform(.05, .12))
    y = my
    if rng.random() < .7:  # heading
        hh = int(ph * rng.uniform(.02, .05)); cv2.rectangle(img, (mx, y), (mx + int((pw - 2 * mx) * rng.uniform(.3, .8)), y + hh), (ink,) * 3, -1); y += hh * 3
    lh = max(3, int(ph * rng.uniform(.012, .022)))
    while y < ph - my - lh:
        if rng.random() < .1: y += lh * 2; continue
        x = mx
        while x < pw - mx - 6:
            wl = int(rng.uniform(.03, .16) * pw); x2 = min(pw - mx, x + wl)
            cv2.rectangle(img, (x, y), (x2, y + lh // 2 + 1), (ink + rng.uniform(0, 30),) * 3, -1); x = x2 + int(pw * .015)
            if rng.random() < .04: break
        y += int(lh * rng.uniform(1.8, 2.6))
    if rng.random() < .3:  # table / photo box
        x0, y0 = mx, int(ph * rng.uniform(.4, .65)); x1, y1 = pw - mx, y0 + int(ph * rng.uniform(.12, .25))
        if rng.random() < .5: cv2.rectangle(img, (x0, y0), (x1, y1), (ink,) * 3, 2)
        else: img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * .5 + rng.uniform(60, 200, 3) * .5
    if rng.random() < .25:  # stamp / logo
        cv2.circle(img, (int(pw * rng.uniform(.2, .8)), int(ph * rng.uniform(.15, .85))), int(pw * rng.uniform(.05, .1)), tuple(float(v) for v in rng.uniform(40, 200, 3)), 3)
    img += (rng.random((ph, pw, 1), dtype=np.float32) - .5) * rng.uniform(1, 6)
    return np.clip(img, 0, 255)


def skin(rng):
    return np.array([rng.uniform(190, 245), rng.uniform(140, 200), rng.uniform(105, 170)]) * rng.uniform(.45, 1.0)


def sample(seed):
    rng = np.random.default_rng(seed)
    aspect = rng.choice([3 / 4, 4 / 3, 9 / 16, 1.0, 2 / 3], p=[.5, .12, .18, .1, .1])
    L = 256; H, W = (L, int(L * aspect)) if aspect <= 1 else (int(L / aspect), L)
    img = background(H, W, rng)
    diag = math.hypot(W, H)
    has_page = rng.random() > .03
    quad = None; mask = np.zeros((H, W), np.uint8)
    if has_page:
        ar = rng.choice([1.414, 1.414, 1.3, 1.0, 1.7, 2.4], p=[.45, .2, .1, .08, .1, .07])
        if rng.random() < .5: ar = 1 / ar  # landscape sheet
        frac = rng.uniform(.35, .93); area = frac * W * H
        ph_ = math.sqrt(area * ar); pw_ = area / ph_
        sc = min(1.0, .97 * min(W / pw_, H / ph_)) if rng.random() < .9 else 1.0
        pw_, ph_ = pw_ * sc, ph_ * sc
        cx, cy = W / 2 + rng.uniform(-.12, .12) * W, H / 2 + rng.uniform(-.12, .12) * H
        ang = rng.normal(0, .22) if rng.random() < .8 else rng.uniform(-math.pi / 2, math.pi / 2)
        pj = rng.uniform(0, .13) if rng.random() < .75 else 0.0  # perspective strength
        corners = np.array([[-pw_ / 2, -ph_ / 2], [pw_ / 2, -ph_ / 2], [pw_ / 2, ph_ / 2], [-pw_ / 2, ph_ / 2]])
        corners = corners + rng.uniform(-1, 1, (4, 2)) * np.array([pw_, ph_]) * pj
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        quad = (corners @ R.T + [cx, cy]).astype(np.float32)
        pimg_w, pimg_h = 360, int(360 * ph_ / max(pw_, 1))
        pimg_h = int(np.clip(pimg_h, 200, 900))
        paper = paper_sheet(pimg_w, pimg_h, rng)
        # lighting gradient over the sheet
        gy, gx = np.mgrid[0:pimg_h, 0:pimg_w].astype(np.float32)
        a = rng.uniform(0, 2 * math.pi); g = (np.cos(a) * gx / pimg_w + np.sin(a) * gy / pimg_h); paper *= (1 + rng.uniform(-.16, .08) * (g[..., None] - g.mean()))
        src = np.array([[0, 0], [pimg_w, 0], [pimg_w, pimg_h], [0, pimg_h]], np.float32)
        M = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(paper, M, (W, H), flags=cv2.INTER_LINEAR)
        mk = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(mk, (quad * 16).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
        mf = mk.astype(np.float32)[..., None] / 255
        mask = mk
        if rng.random() < .75:  # soft shadow of the sheet on the surface
            dx, dy = rng.uniform(-14, 14), rng.uniform(2, 16); sh = cv2.warpAffine(mf[..., 0], np.float32([[1, 0, dx], [0, 1, dy]]), (W, H))
            sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(3, 14))[..., None]; img = img * (1 - rng.uniform(.1, .42) * sh)
        img = img * (1 - mf) + warped * mf
        if rng.random() < .25:  # hand / object shadow across the page
            ov = np.zeros((H, W), np.float32); cv2.ellipse(ov, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), (int(rng.uniform(25, 90)), int(rng.uniform(15, 60))), rng.uniform(0, 180), 0, 360, 1, -1)
            ov = cv2.GaussianBlur(ov, (0, 0), rng.uniform(8, 22))[..., None]; img = img * (1 - rng.uniform(.1, .3) * ov)
        if rng.random() < .4:  # fingers holding the sheet
            for _ in range(int(rng.integers(1, 4))):
                i = int(rng.integers(0, 4)); p0, p1 = quad[i], quad[(i + 1) % 4]; t = rng.uniform(.1, .9); pt = p0 + (p1 - p0) * t
                cen = np.array([W / 2, H / 2]); out = pt - cen; out = out / (np.linalg.norm(out) + 1e-6); L1 = rng.uniform(22, 55); ctr = pt + out * rng.uniform(-6, L1 * .5)
                ov = np.zeros((H, W), np.float32); cv2.ellipse(ov, (int(ctr[0]), int(ctr[1])), (int(L1), int(rng.uniform(9, 17))), math.degrees(math.atan2(out[1], out[0])), 0, 360, 1, -1)
                ov = cv2.GaussianBlur(ov, (0, 0), 1.2)[..., None]; img = img * (1 - ov) + skin(rng) * ov * rng.uniform(.9, 1.05)
    if rng.random() < .25:  # unrelated objects on the surface
        for _ in range(int(rng.integers(1, 3))):
            c = (rng.uniform(0, W), rng.uniform(0, H)); sz = (rng.uniform(15, 70), rng.uniform(8, 40)); box = cv2.boxPoints((c, sz, rng.uniform(0, 180))).astype(np.int32)
            ov = np.zeros((H, W), np.float32); cv2.fillConvexPoly(ov, box, 1.0); ov = cv2.GaussianBlur(ov, (0, 0), .8)[..., None]
            img = img * (1 - ov * .9) + rand_color(rng, str(rng.choice(['dark', 'mid', 'sat']))) * ov * .9
    if rng.random() < .15:  # pen
        p0 = (int(rng.uniform(0, W)), int(rng.uniform(0, H))); a = rng.uniform(0, 6.28); p1 = (int(p0[0] + math.cos(a) * rng.uniform(40, 110)), int(p0[1] + math.sin(a) * rng.uniform(40, 110)))
        cv2.line(img, p0, p1, tuple(float(v) for v in rand_color(rng, str(rng.choice(['dark', 'sat'])))), int(rng.integers(2, 5)), cv2.LINE_AA)
    # camera: exposure, colour cast, vignette, blur, noise, jpeg
    img = img * rng.uniform(.75, 1.2) + rng.uniform(-18, 18); img = img * rng.uniform(.93, 1.07, 3)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); v = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2; img = img * (1 - rng.uniform(0, .5) * v)[..., None]
    img = np.clip(img, 0, 255)
    if rng.random() < .7: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(.3, 1.8))
    if rng.random() < .1:
        k = int(rng.integers(5, 13)); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1 / k; M2 = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1); img = cv2.filter2D(img, -1, cv2.warpAffine(ker, M2, (k, k)))
    img = img + rng.normal(0, rng.uniform(0, 7), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 96))]); img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    x = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA); m = cv2.resize(mask, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    q = None if quad is None else (quad / np.array([W, H], np.float32)).astype(np.float32)  # normalised corners
    return x, m, q


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    tiles = []
    for i in range(n):
        x, m, q = sample(10_000 + i)
        big = cv2.resize(x, (192, 192)); ov = big.copy(); mm = cv2.resize(m, (192, 192)); ov[..., 1] = np.clip(ov[..., 1] * (1 - .45 * mm) + 255 * .45 * mm, 0, 255).astype(np.uint8)
        tiles.append(np.concatenate([big, ov], 0))
    cols = 8; rows = [np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(n // cols)]
    cv2.imwrite('samples.jpg', cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    import time; t = time.time()
    for i in range(200): sample(i)
    print('ms/sample', round((time.time() - t) / 200 * 1000, 1))
