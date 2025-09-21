# === Inference Script (Single Cell) ===

# 1) Install dependencies (uncomment if needed)
# !pip install -q segmentation-models-pytorch albumentations

import os, gc
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

# 2) Paths (adjust to your local mount)
DATA_ROOT      = "/path/to/aerialimageryforroofsegmentation"
TEST_IMG_DIR   = os.path.join(DATA_ROOT, "test",  "image")
TEST_MASK_DIR  = os.path.join(DATA_ROOT, "test",  "label")
CHECKPOINT     = "/path/to/working_patch1/models_stage1/unet_best.pth"
OUTPUT_DIR     = "/path/to/working_patch1/test_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3) Device and Model load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(device)

# wrap DataParallel on 8 V100s
if torch.cuda.device_count() > 1:
    print(f"[INFO] DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

chk = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(chk)
model.eval()

# 4) Pre‐processing transform
infer_tf = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

# 5) Patch‐based inference
def infer_large(img_np, patch_size=401, overlap=64, thr=0.5):
    H, W, _ = img_np.shape
    stride = patch_size - overlap
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y+patch_size, H)
            x1 = min(x+patch_size, W)
            patch = img_np[y:y1, x:x1]
            ph, pw, _ = patch.shape

            canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            canvas[:ph, :pw] = patch

            aug = infer_tf(image=canvas)
            inp = aug["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            probs = probs[:ph, :pw]
            prob_map[y:y1, x:x1] += probs
            count_map[y:y1, x:x1] += 1.0

            del inp, logits, probs
            torch.cuda.empty_cache()

    avg = prob_map / (count_map + 1e-8)
    return (avg > thr).astype(np.uint8)

# 6) Gather test files
img_files = sorted(glob(os.path.join(TEST_IMG_DIR, "*.*")))
mask_map  = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in glob(os.path.join(TEST_MASK_DIR, "*.*"))
}

# 7) Metrics accumulators
ious, f1s, precs, recs, accs = [], [], [], [], []

# 8) Run inference + evaluation
for img_path in tqdm(img_files, desc="Inference"):
    key = os.path.splitext(os.path.basename(img_path))[0]
    if key not in mask_map:
        continue

    img_np = np.array(Image.open(img_path).convert("RGB"))
    gt     = np.array(Image.open(mask_map[key]).convert("L"))
    gt     = (gt > 0).astype(np.uint8)

    pred = infer_large(img_np, patch_size=401, overlap=64, thr=0.5)

    # save mask
    outp = os.path.join(OUTPUT_DIR, f"{key}_pred.png")
    Image.fromarray((pred*255).astype(np.uint8)).save(outp)

    # compute metrics
    p_flat = pred.reshape(-1)
    g_flat = gt.reshape(-1)
    ious .append(jaccard_score(   g_flat, p_flat, zero_division=0))
    f1s  .append(f1_score(       g_flat, p_flat, zero_division=0))
    precs.append(precision_score(g_flat, p_flat, zero_division=0))
    recs .append(recall_score(   g_flat, p_flat, zero_division=0))
    accs .append((p_flat==g_flat).mean())

    gc.collect()

# 9) Print overall performance
print("\n=== Test Metrics ===")
print(f"IoU:  {np.mean(ious):.4f} ± {np.std(ious):.4f}")
print(f"F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Prec: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Rec:  {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")

print(f"\nSaved predictions to {OUTPUT_DIR}")