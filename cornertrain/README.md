# cornertrain — fine-tuning the page-corner network

The scanner app finds the four corners of a page with a small neural network. This folder holds the scripts that
adapt that network to real photos. It is the second model project of the Scanner app; the first one is the rest of this repository
(*pagequad*, trained on synthetic images only).

## Why this exists

| Step | What happened | Lesson |
|---|---|---|
| 1. Classic detector only (OpenCV: edges, Otsu, contours) | Good on a dark table, wrong when the page and the wall/cloth behind it have the same colour | Rules cannot tell "this is a page" from "this is another light rectangle" |
| 2. `docmodel/pagequad` (own U-Net, 56 k parameters, **synthetic** training images) | Worked on synthetic scenes, called 81 % of a real cluttered photo "paper" | A model trained only on synthetic scenes does not transfer to real photos |
| 3. Third-party corner network (KennethTM/document_corner_detector, MIT, MobileNetV3-small U-Net, 3.6 M parameters, ONNX) | Right on 3 of our 4 real test photos, but on a 877-image held-out set: median worst-corner error 2.9 % of the frame diagonal, **36 % of images with a corner more than 10 % off** | Better, still not good enough, and its training data is unknown |
| 4. **This folder**: fine-tune that network on DocCornerDataset (real photos with corner labels) | Same held-out set: median 0.84 %, failures >10 % down to **13 %**; all 4 of our real photos fit tightly with confidence 0.8–0.9 | The fix was real training data, not more rules |

## What was changed, and how

1. **`convert.py` — ONNX → PyTorch.** The original model only exists as an ONNX file. `onnx2torch` converts it; two
   `Clip` nodes with an empty min/max input had to be patched first. The converted module reproduces the ONNX output
   to 1e-6, so training starts exactly from the published weights. (Do **not** save the start file after a test
   training step: an early run saved weights after two steps on random noise and gave a bad baseline.)
2. **`prep.py` — data.** Reads DocCornerDataset parquet shards, keeps all paper / receipt / book / invoice images, one
   in five ID cards and a quarter of the negatives (no document), resizes to ≤ 448 px, stores one `.npy`.
   7 train shards = 4 093 images; 1 test shard = 577 images (held out, never trained on).
   Only numpy/cv2/pyarrow in the worker processes (no torch), to keep memory low.
3. **`train.py` — fine-tuning.**
   * Input: image letter-boxed into 224×224 with black padding (the app does the same); ImageNet mean/std.
   * Target: 4 gaussian heatmaps (σ = 3 px), one per corner in the order TL, TR, BR, BL (geometric: TL has the
     smallest x+y …). The dataset order is geometric in 100 % of the samples, so rotations by 90° and flips are
     allowed; corners are re-sorted after every augmentation.
   * Augmentations: 90° rotations, flips, small rotation and scale, random crop around the page (page small or
     filling the frame), brightness/contrast/gamma, blur, noise, random plain borders.
   * Loss: `mean((pred − heat)² · (1 + 30·heat)) · 100` — plain MSE would learn to output zeros because the corner is a
     few pixels of 50 000.
   * AdamW, lr 2–3e-4 with cosine decay, batch 16, 10–17 epochs on CPU (about 4–7 min/epoch).
   * Validation metric: the worst of the 4 corners, divided by the frame diagonal; reports median, share < 3 % and
     share > 10 %. The best epoch by (median + share > 10 %) is kept.
4. **Export.** `torch.onnx.export(..., opset_version=17, dynamo=False)` → `corners2.onnx`, same input/output names as
   the original (`input` 1×3×224×224, `output` 1×4×224×224), maximum difference to the PyTorch module 9e-7. The file
   is copied to `app/corners.onnx`.

## How the app uses the network (see `app/index.html` of the Scanner app, `CornerNet`)

* Runs with onnxruntime-web (wasm, one thread), offline, about 340 ms per run on the test phone (Galaxy S23).
* **Top-3 peaks per corner**, then the combination with the highest total confidence minus a penalty for not being a
  parallelogram in perspective. A single stray second peak used to drag one corner across the frame.
* **Photo**: two passes (whole frame, then a zoomed crop around the first answer). The result is compared with the
  classical detector: if the worst corner differs by more than 4 % of the diagonal the network wins; if they agree the
  classical outline is kept because it also measures the bend of the sheet.
* **Live preview**: one pass per detection, median of the last 3 results, and in between the outline is carried by
  Lucas–Kanade optical flow of a 3×3 grid inside the page (a homography updates the 4 corners every ~30 ms), then
  eased on screen every frame. The live outline is always straight; bend is measured on the real photo.

## Things that were tried and did not work (so they are not repeated)

* WebGL / WebGPU execution provider: the exported graph uses `int64` shape ops (removed with `onnxsim`), then WebGL
  still rejects `Resize` mode `nearest`; WebGPU is not available in this WebView. CPU wasm only.
* Pulling each side of the outline in to the text ("margin check"), a "text region" candidate, wider search bands
  around the classical contour: all broke the handwritten notebook photo or changed nothing.
* Checking the network by shrinking a photo into a dark frame: the network then finds the frame border. Not a valid test.

## Licences and data

* Network architecture and original weights: MIT (KennethTM/document_corner_detector, Hugging Face Space).
* onnxruntime-web: MIT.
* DocCornerDataset: card says `license: other`. Sources: MIDV-500 and MIDV-2019 (research use), SmartDoc (research
  use), COCO (CC BY 4.0, negatives), several Roboflow Universe datasets (licences vary). **Check these before
  distributing the app or the trained weights.** The trained weights inherit that uncertainty.

## Reproduce

```bash
python convert.py original.onnx start.pt
python prep.py "train/d*.parquet" train.npy
python prep.py "test/data-00000.parquet" val.npy
python train.py train.npy val.npy 10 best.pt            # from start.pt
python train.py train.npy val.npy 7 best2.pt best.pt    # continue from a checkpoint
```
Data lives outside this folder (parquet shards are ~200 MB each). `start.pt`, `best*.pt`, `*.npy` and logs are not
committed.

## Results

Held-out test shard (577 images, worst corner / frame diagonal):

| Model | median | < 3 % | > 10 % |
|---|---|---|---|
| original (KennethTM) | 2.9 % | 51 % | 36 % |
| epoch 0 of fine-tuning | 1.0 % | 69 % | 18 % |
| final (epoch 6 of the second run) | 0.84 % | 74 % | 13 % |

Own photos (notebook ×2, book on table, book on lap): original confidences 0.06–0.5 with one wrong corner;
fine-tuned 0.6–0.9 and all corners on the page.

## Next ideas

* More shards (only 7 of ~33 train shards were used) and fine-tuning on the user's own labelled photos on white cloth
  and other light backgrounds, which the dataset covers poorly.
* A smaller / quantised model or `numThreads > 1` (needs cross-origin isolation in the WebView) for a faster live preview.
