# train.py

# === Install & Imports ===
# (Uncomment if running in a fresh environment)
# !pip install -q segmentation-models-pytorch albumentations

import os
import gc
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

# === Paths & Checkpoint Setup ===
PROJECT_ROOT = "."
DATA_DIR     = os.path.join(PROJECT_ROOT, "aerialimageryforroofsegmentation")

AIRS_TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train", "image")
AIRS_TRAIN_MASK_DIR  = os.path.join(DATA_DIR, "train", "label")
AIRS_VAL_IMAGE_DIR   = os.path.join(DATA_DIR, "val",   "image")
AIRS_VAL_MASK_DIR    = os.path.join(DATA_DIR, "val",   "label")

AIRS_TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
AIRS_VAL_TXT   = os.path.join(DATA_DIR, "val_modified.txt")

SAVE_DIR         = os.path.join(PROJECT_ROOT, "working_patch1", "models_stage1")
VISUAL_DIR       = os.path.join(PROJECT_ROOT, "working_patch1", "visualizations")
TRAIN_VISUAL_DIR = os.path.join(VISUAL_DIR, "train")
VAL_VISUAL_DIR   = os.path.join(VISUAL_DIR, "val")
CHECKPOINT_PATH  = os.path.join(SAVE_DIR, "checkpoint.pth")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRAIN_VISUAL_DIR, exist_ok=True)
os.makedirs(VAL_VISUAL_DIR, exist_ok=True)

# === Multi-GPU Setup: try DDP, else DataParallel/single-GPU ===
use_ddp = False
try:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"[DDP] rank {rank}/{world_size} on GPU {local_rank}")
    use_ddp = True
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] DDP not initialized, using device: {device}")

# === Dataset Definition ===
class RooftopDataset(Dataset):
    def __init__(self, image_dir, mask_dir, list_file, patch_size, transforms=None):
        with open(list_file, 'r') as f:
            names = [l.strip() for l in f]
        self.image_paths = [os.path.join(image_dir, n) for n in names]
        self.mask_paths  = [os.path.join(mask_dir,  n) for n in names]
        self.ps = patch_size
        self.tf = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        msk = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        msk = (msk > 0).astype(np.uint8)
        H, W, _ = img.shape

        # Random 401×401 patch with at least one mask pixel
        for _ in range(10):
            top  = np.random.randint(0, H - self.ps + 1)
            left = np.random.randint(0, W - self.ps + 1)
            img_p = img[top:top+self.ps, left:left+self.ps]
            msk_p = msk[top:top+self.ps, left:left+self.ps]
            if msk_p.sum() > 0:
                img, msk = img_p, msk_p
                break
        else:
            ct, cl = (H-self.ps)//2, (W-self.ps)//2
            img = img[ct:ct+self.ps, cl:cl+self.ps]
            msk = msk[ct:ct+self.ps, cl:cl+self.ps]

        if self.tf:
            aug = self.tf(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            msk = torch.from_numpy(msk).unsqueeze(0).float()

        return img, msk

# === Transforms ===
train_tf = A.Compose([
    A.RandomCrop(401, 401),
    A.HorizontalFlip(0.5),
    A.VerticalFlip(0.5),
    A.RandomRotate90(0.5),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ToTensorV2(),
])
val_tf = A.Compose([
    A.CenterCrop(401, 401),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ToTensorV2(),
])

# === DataLoaders ===
from torch.utils.data.distributed import DistributedSampler

patch_size = 401
train_ds = RooftopDataset(AIRS_TRAIN_IMAGE_DIR, AIRS_TRAIN_MASK_DIR,
                          AIRS_TRAIN_TXT, patch_size, transforms=train_tf)
val_ds   = RooftopDataset(AIRS_VAL_IMAGE_DIR, AIRS_VAL_MASK_DIR,
                          AIRS_VAL_TXT,   patch_size, transforms=val_tf)

if use_ddp:
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)
    shuffle_train = False
else:
    train_sampler = None
    val_sampler   = None
    shuffle_train = True

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=shuffle_train,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    sampler=val_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# === Model & Parallel Wrapping ===
def get_unet():
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_dropout=0.2
    )

model_single = get_unet().to(device)
if use_ddp:
    model = DDP(model_single, device_ids=[local_rank], output_device=local_rank)
elif torch.cuda.device_count() > 1:
    model = nn.DataParallel(model_single)
else:
    model = model_single

# === Loss & Optimizer ===
dice  = smp.losses.DiceLoss(mode="binary", from_logits=True)
focal = smp.losses.FocalLoss(mode="binary")
class ComboLoss(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.d, self.f = d, f
    def forward(self, x, y):
        return self.d(x, y) + self.f(x, y)

loss_fn   = ComboLoss(dice, focal)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
scaler    = torch.cuda.amp.GradScaler()

# === Training Loop ===
best_iou = 0.0
num_epochs = 30

for epoch in range(1, num_epochs+1):
    if use_ddp: train_loader.sampler.set_epoch(epoch)
    model.train()
    tloss = 0.0
    for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outs = model(imgs)
            loss = loss_fn(outs, msks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tloss += loss.item() * imgs.size(0)
    tloss /= len(train_loader.dataset)
    print(f"[TRAIN] Epoch {epoch} Loss: {tloss:.4f}")

    model.eval()
    vloss = 0.0; tp=fp=fn=tn=0
    with torch.no_grad():
        for imgs, msks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs, msks = imgs.to(device), msks.to(device)
            with torch.cuda.amp.autocast():
                outs = model(imgs)
                loss = loss_fn(outs, msks)
            vloss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(outs)>0.5).float().view(-1)
            gts   = msks.view(-1)
            tp += (preds * gts).sum().item()
            fp += (preds * (1-gts)).sum().item()
            fn += ((1-preds) * gts).sum().item()
            tn += ((1-preds)*(1-gts)).sum().item()

    vloss /= len(val_loader.dataset)
    eps = 1e-7
    iou  = tp/(tp+fp+fn+eps)
    acc  = (tp+tn)/(tp+tn+fp+fn+eps)
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    f1   = 2*tp/(2*tp+fp+fn+eps)

    print(f"[VAL] Epoch {epoch} Loss: {vloss:.4f} IoU: {iou:.4f} F1: {f1:.4f} "
          f"Prec: {prec:.4f} Rec: {rec:.4f} Acc: {acc:.4f}")

    scheduler.step(iou)
    if iou > best_iou:
        best_iou = iou
        save_path = CHECKPOINT_PATH
        if use_ddp: torch.save(model.module.state_dict(), save_path)
        else:      torch.save(model.state_dict(),       save_path)
        print(f"[SAVE] New best IoU: {best_iou:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

print(f"\nTraining complete, best IoU: {best_iou:.4f}")