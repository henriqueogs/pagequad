"""Decode DocCornerDataset parquet shards into one compact npz: images resized to <=448 px (longest side), corners in pixels.
Keeps paper / receipt / book / invoice style documents, a share of ID cards, and negatives. Uses numpy+cv2+pyarrow only (no torch) so worker processes stay light."""
import sys, glob, zlib, numpy as np, cv2, pyarrow.parquet as pq
from multiprocessing import Pool
S = 448
ID_KEYS = ('midv', 'card', 'id_', 'ktp', 'passport')
def load(path):
    out = []
    for r in pq.read_table(path).to_pylist():
        name = r['filename']; is_id = any(k in name for k in ID_KEYS)
        if r['is_negative']:
            keep = zlib.crc32(name.encode()) % 4 == 0   # a quarter of the negatives
        else:
            keep = (not is_id) or zlib.crc32(name.encode()) % 5 == 0   # all documents, one in five ID cards
        if not keep: continue
        im = cv2.imdecode(np.frombuffer(r['image']['bytes'], np.uint8), 1)
        if im is None: continue
        h, w = im.shape[:2]; k = min(1.0, S / max(h, w))
        if k < 1: im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        h2, w2 = im.shape[:2]
        c = None if r['is_negative'] else np.array([[r[f'corner_{p}_x'] * w2, r[f'corner_{p}_y'] * h2] for p in ('tl', 'tr', 'br', 'bl')], np.float32)
        out.append((cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), c, name))
    return out
if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(src))
    with Pool(4) as p: res = p.map(load, files)
    items = [x for r in res for x in r]
    np.save(dst, np.array([(i[0], i[1], i[2]) for i in items], dtype=object), allow_pickle=True)
    print(len(items), 'items;', sum(i[1] is None for i in items), 'negatives')
