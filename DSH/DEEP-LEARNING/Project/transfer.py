import argparse, json, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ["scratch", "feature_extraction", "finetune_last", "finetune_all"]


def get_data(args):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tr = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    if args.data_dir:
        train = datasets.ImageFolder(os.path.join(args.data_dir, "train"), tr)
        val = datasets.ImageFolder(os.path.join(args.data_dir, "val"), te)
        n_cls = len(train.classes)
    else:  # Flowers102: 1020 train / 1020 val / 6149 test images, 102 classes
        train = datasets.Flowers102(args.root, split="train", transform=tr, download=True)
        val = datasets.Flowers102(args.root, split="test", transform=te, download=True)
        n_cls = 102
    mk = lambda ds, sh: DataLoader(ds, args.bs, shuffle=sh, num_workers=args.workers, pin_memory=True)
    return mk(train, True), mk(val, False), n_cls


def build_model(name, strategy, n_cls):
    m = models.get_model(name, weights=None if strategy == "scratch" else "DEFAULT")
    # replace classification head
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, n_cls); head = list(m.fc.parameters())
        last_block = list(m.layer4.parameters())
    else:
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls); head = list(m.classifier[-1].parameters())
        last_block = list(m.features[24:].parameters()) + list(m.classifier[:-1].parameters())
    head_ids = {id(p) for p in head}; last_ids = {id(p) for p in last_block}
    backbone = [p for p in m.parameters() if id(p) not in head_ids]

    if strategy == "feature_extraction":
        for p in backbone: p.requires_grad = False
    elif strategy == "finetune_last":
        for p in backbone:
            p.requires_grad = id(p) in last_ids
    return m, head, backbone


def freeze_bn(model):
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d): mod.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = top5 = n = 0; loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device); out = model(x)
        loss += nn.functional.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        top5 += (out.topk(5, 1).indices == y[:, None]).any(1).sum().item(); n += len(y)
    return loss / n, correct / n, top5 / n


def run(strategy, args, tl, vl, n_cls, device):
    model, head, backbone = build_model(args.model, strategy, n_cls); model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = [{"params": head, "lr": args.lr}]
    bb = [p for p in backbone if p.requires_grad]
    if bb: groups.append({"params": bb, "lr": args.lr if strategy == "scratch" else args.lr * 0.1})
    opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}; t0 = time.time()
    print(f"\n=== {args.model} | {strategy} | trainable params: {trainable/1e6:.2f}M ===")
    for ep in range(1, args.epochs + 1):
        model.train()
        if strategy == "feature_extraction": freeze_bn(model)
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        vloss, acc, acc5 = evaluate(model, vl, device)
        hist["train_loss"].append(tot / len(tl)); hist["val_loss"].append(vloss); hist["val_acc"].append(acc)
        print(f"ep {ep:02d} | train {hist['train_loss'][-1]:.3f} | val {vloss:.3f} | top1 {acc:.3f} | top5 {acc5:.3f}")
    return {"final_top1": acc, "final_top5": acc5, "best_top1": max(hist["val_acc"]),
            "trainable_params_M": trainable / 1e6, "train_time_s": time.time() - t0, "history": hist}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18", help="any torchvision classifier with .fc or VGG-style .classifier")
    p.add_argument("--data_dir", default=None); p.add_argument("--root", default="./data")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    p.add_argument("--out", default="outputs_b")
    # Use parse_known_args() so that Jupyter's automatic backend arguments (-f) are safely ignored
    a, _ = p.parse_known_args(); os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tl, vl, n_cls = get_data(a)
    results = {s: run(s, a, tl, vl, n_cls, device) for s in a.strategies}
    json.dump(results, open(f"{a.out}/{a.model}_results.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for s, r in results.items():
        ax[0].plot(r["history"]["val_acc"], label=s); ax[1].plot(r["history"]["train_loss"], label=s)
    ax[0].set_title("Validation top-1 accuracy"); ax[1].set_title("Training loss"); ax[0].legend()
    ax[2].bar(list(results), [r["final_top1"] for r in results.values()]); ax[2].set_title("Final top-1 accuracy")
    ax[2].tick_params(axis="x", rotation=25)
    plt.tight_layout(); plt.savefig(f"{a.out}/{a.model}_comparison.png", dpi=130)
    print("\n{:<20}{:>10}{:>10}{:>14}{:>12}".format("strategy", "top1", "top5", "trainable(M)", "time(s)"))
    for s, r in results.items():
        print("{:<20}{:>10.3f}{:>10.3f}{:>14.2f}{:>12.0f}".format(s, r["final_top1"], r["final_top5"], r["trainable_params_M"], r["train_time_s"]))


if __name__ == "__main__":
    main()
