"""
Part (a): U-Net for semantic segmentation on Pascal VOC 2012.
Metrics: mean IoU, per-class IoU, pixel accuracy.  Visualises image / ground truth / prediction.

Run:  python part_a_unet.py --root ./data --epochs 30 --img_size 256
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_CLASSES = 21  # 20 objects + background
IGNORE = 255      # VOC "void" boundary label
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- model
class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Classic U-Net: 4 down-sampling stages, bottleneck, 4 up-sampling stages with skip connections."""
    def __init__(self, in_ch=3, n_classes=NUM_CLASSES, base=64, dropout=0.2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = DoubleConv(in_ch, c[0])
        self.downs = nn.ModuleList([DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.up_convs = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.outc = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(self.pool(skips[-1])))
        x = self.drop(skips.pop())                      # bottleneck
        for up, conv in zip(self.ups, self.up_convs):
            x = up(x)
            x = conv(torch.cat([skips.pop(), x], dim=1))  # skip connection
        return self.outc(x)


# ----------------------------------------------------------------------------- data
class VOCSeg(torch.utils.data.Dataset):
    def __init__(self, root, split, size, train, download):
        self.ds = VOCSegmentation(root, year="2012", image_set=split, download=download)
        self.size, self.train = size, train

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        img, mask = self.ds[i]
        img = TF.resize(img, [self.size, self.size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], InterpolationMode.NEAREST)
        if self.train:
            if torch.rand(1) < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            img = TF.adjust_brightness(img, 0.8 + 0.4 * torch.rand(1).item())
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


# ----------------------------------------------------------------------------- metrics
def confusion(pred, target, n=NUM_CLASSES):
    valid = target != IGNORE
    idx = target[valid] * n + pred[valid]
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def iou_from_conf(conf):
    conf = conf.double()
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / union.clamp(min=1)
    iou[union == 0] = float("nan")                      # class absent from data
    pix_acc = (inter.sum() / conf.sum()).item()
    return iou.cpu().numpy(), float(np.nanmean(iou.cpu().numpy())), pix_acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, ignore_index=IGNORE).item() * x.size(0)
        n += x.size(0)
        conf += confusion(out.argmax(1), y)
    iou, miou, acc = iou_from_conf(conf)
    return loss_sum / n, iou, miou, acc


# ----------------------------------------------------------------------------- visualisation
def voc_cmap(N=256):
    def bit(v, i): return (v >> i) & 1
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0; c = i
        for j in range(8):
            r |= bit(c, 0) << (7 - j); g |= bit(c, 1) << (7 - j); b |= bit(c, 2) << (7 - j); c >>= 3
        cmap[i] = [r, g, b]
    return cmap


@torch.no_grad()
def visualize(model, dataset, device, path, n=6):
    model.eval(); cmap = voc_cmap()
    idxs = np.linspace(0, len(dataset) - 1, n).astype(int)
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    for r, i in enumerate(idxs):
        x, y = dataset[i]
        pred = model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        for c, (im, t) in enumerate([(img, "Image"), (cmap[y.numpy()], "Ground truth"), (cmap[pred], "U-Net prediction")]):
            ax[r, c].imshow(im); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(t)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data"); p.add_argument("--out", default="outputs_a")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base", type=int, default=64); p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    train_ds = VOCSeg(a.root, "train", a.img_size, True, download=True)
    val_ds = VOCSeg(a.root, "val", a.img_size, False, download=False)
    tl = DataLoader(train_ds, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(val_ds, a.bs, shuffle=False, num_workers=a.workers)

    model = UNet(base=a.base).to(device)
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    hist, best = {"train_loss": [], "val_loss": [], "val_miou": [], "val_acc": []}, 0.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot, t0 = 0.0, time.time()
        for x, y in tqdm(tl, desc=f"epoch {ep}/{a.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device if device != "mps" else "cpu", enabled=use_amp):
                loss = F.cross_entropy(model(x), y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        vloss, iou, miou, acc = evaluate(model, vl, device)
        hist["train_loss"].append(tot / len(tl)); hist["val_loss"].append(vloss)
        hist["val_miou"].append(miou); hist["val_acc"].append(acc)
        print(f"ep {ep:02d} | train {hist['train_loss'][-1]:.3f} | val {vloss:.3f} | mIoU {miou:.3f} | pixAcc {acc:.3f} | {time.time()-t0:.0f}s")
        if miou > best:
            best = miou; torch.save(model.state_dict(), f"{a.out}/unet_best.pt")

    model.load_state_dict(torch.load(f"{a.out}/unet_best.pt", map_location=device))
    _, iou, miou, acc = evaluate(model, vl, device)
    print(f"\nBEST  mIoU={miou:.4f}  pixel-acc={acc:.4f}")
    json.dump({"mIoU": miou, "pixel_acc": acc, "per_class_IoU": dict(zip(VOC_CLASSES, map(float, iou))),
               "history": hist}, open(f"{a.out}/metrics.json", "w"), indent=2)

    # curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["val_miou"], label="mIoU"); ax[1].plot(hist["val_acc"], label="pixel acc"); ax[1].set_title("Validation"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/curves.png", dpi=130); plt.close()
    # per-class IoU bar chart
    plt.figure(figsize=(10, 3.5)); plt.bar(VOC_CLASSES, np.nan_to_num(iou)); plt.xticks(rotation=70)
    plt.ylabel("IoU"); plt.title(f"Per-class IoU (mIoU={miou:.3f})"); plt.tight_layout()
    plt.savefig(f"{a.out}/per_class_iou.png", dpi=130); plt.close()
    visualize(model, val_ds, device, f"{a.out}/predictions.png")
    print("Saved results to", a.out)


if __name__ == "__main__":
    main()
