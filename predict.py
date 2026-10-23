"""Find the sheet of paper in a photo.

    python predict.py photo.jpg [--weights net.pt] [--out overlay.jpg]

Prints the four corners (top-left, top-right, bottom-right, bottom-left) in pixels of the original image and writes an overlay.
The network only produces a probability mask; the corners come from the mask with plain geometry (see train.quad_from_mask).
"""
import argparse
import numpy as np
import cv2
import torch
from model import Net
from train import quad_from_mask
from gen import S


def predict(img_bgr, net):
    h, w = img_bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (S, S), interpolation=cv2.INTER_AREA)  # same squash as in training
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255))[0, 0].numpy()
    q = quad_from_mask(p)
    return p, None if q is None else q * np.array([w, h], np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--weights', default='net.pt'); ap.add_argument('--out', default='overlay.jpg')
    a = ap.parse_args()
    net = Net(); net.load_state_dict(torch.load(a.weights)); net.eval()
    img = cv2.imread(a.image)
    p, q = predict(img, net)
    if q is None:
        print('no sheet found')
    else:
        print('corners (px):', np.round(q, 1).tolist())
        vis = img.copy(); cv2.polylines(vis, [q.astype(np.int32)], True, (0, 220, 0), max(2, img.shape[1] // 200), cv2.LINE_AA)
        mask = cv2.resize(p, (img.shape[1], img.shape[0]))[..., None]; vis = (vis * (1 - .25 * mask) + np.array([0, 255, 0]) * .25 * mask).astype(np.uint8)
        cv2.imwrite(a.out, vis); print('overlay ->', a.out)
