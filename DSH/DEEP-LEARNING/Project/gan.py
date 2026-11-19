import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZDIM = 100


# ----------------------------------------------------------------------------- models
class MLPG(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(ZDIM, 256), nn.LeakyReLU(0.2), nn.Linear(256, 512), nn.LeakyReLU(0.2),
                              nn.Linear(512, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 784), nn.Tanh())
    def forward(s, z): return s.net(z).view(-1, 1, 28, 28)

class MLPD(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                              nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(s, x): return s.net(x).view(-1)

class ConvG(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.ConvTranspose2d(ZDIM, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),   # 7x7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),    # 14x14
            nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())                                          # 28x28
    def forward(s, z): return s.net(z.view(-1, ZDIM, 1, 1))

class ConvD(nn.Module):
    def __init__(s, bn=True):
        super().__init__()
        norm = (lambda c: nn.BatchNorm2d(c)) if bn else (lambda c: nn.Identity())
        s.net = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
                              nn.Conv2d(64, 128, 4, 2, 1), norm(128), nn.LeakyReLU(0.2, True),
                              nn.Flatten(), nn.Linear(128 * 49, 1))
    def forward(s, x): return s.net(x).view(-1)

def init_w(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)): nn.init.normal_(m.weight, 0, 0.02)


# ----------------------------------------------------------------------------- evaluation classifier
class Classifier(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                               nn.Linear(64 * 49, 128), nn.ReLU())
        s.fc = nn.Linear(128, 10)
    def forward(s, x):
        f = s.conv(x); return s.fc(f), f

def train_classifier(loader, device, epochs=3):
    clf = Classifier().to(device); opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(clf(x)[0], y); opt.zero_grad(); loss.backward(); opt.step()
    return clf.eval()

@torch.no_grad()
def feats_probs(clf, imgs, device, bs=500):
    fs, ps = [], []
    for i in range(0, len(imgs), bs):
        logits, f = clf(imgs[i:i + bs].to(device)); fs.append(f.cpu()); ps.append(F.softmax(logits, 1).cpu())
    return torch.cat(fs).numpy(), torch.cat(ps).numpy()

def inception_score(p, eps=1e-12):
    py = p.mean(0, keepdims=True)
    return float(np.exp((p * (np.log(p + eps) - np.log(py + eps))).sum(1).mean()))

def fid(f1, f2):
    m1, m2 = f1.mean(0), f2.mean(0); s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False); covmean = covmean.real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


# ----------------------------------------------------------------------------- training
def gradient_penalty(D, real, fake, device):
    a = torch.rand(real.size(0), 1, 1, 1, device=device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    g, = torch.autograd.grad(D(x).sum(), x, create_graph=True)
    return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

def train_variant(name, loader, device, args, out):
    if name == "vanilla":   G, D = MLPG(), MLPD()
    elif name == "dcgan":   G, D = ConvG(), ConvD(bn=True)
    else:                   G, D = ConvG(), ConvD(bn=False)   # critic without BN (required for GP)
    G.to(device).apply(init_w); D.to(device).apply(init_w)
    if name == "wgan_gp":
        oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9)); oD = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.0, 0.9))
    else:
        oG = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))
    fixed = torch.randn(64, ZDIM, device=device); real_label = 0.9 if args.label_smooth else 1.0
    losses = {"G": [], "D": []}; os.makedirs(f"{out}/{name}", exist_ok=True); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        gl = dl = 0.0
        for it, (real, _) in enumerate(loader):
            real = real.to(device); b = real.size(0); z = torch.randn(b, ZDIM, device=device)
            fake = G(z)
            if name == "wgan_gp":
                d_loss = D(fake.detach()).mean() - D(real).mean() + 10.0 * gradient_penalty(D, real, fake.detach(), device)
                oD.zero_grad(); d_loss.backward(); oD.step()
                if (it + 1) % 5 == 0:                              # n_critic = 5
                    g_loss = -D(G(torch.randn(b, ZDIM, device=device))).mean()
                    oG.zero_grad(); g_loss.backward(); oG.step()
                else:
                    g_loss = torch.tensor(float("nan"))
            else:
                d_loss = F.binary_cross_entropy_with_logits(D(real), torch.full((b,), real_label, device=device)) + \
                         F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(b, device=device))
                oD.zero_grad(); d_loss.backward(); oD.step()
                g_loss = F.binary_cross_entropy_with_logits(D(fake), torch.ones(b, device=device))   # non-saturating
                oG.zero_grad(); g_loss.backward(); oG.step()
            dl += d_loss.item()
            if not torch.isnan(g_loss): gl += g_loss.item()
        n_g = len(loader) // 5 if name == "wgan_gp" else len(loader)
        losses["D"].append(dl / len(loader)); losses["G"].append(gl / max(n_g, 1))
        print(f"[{name}] ep {ep:02d} | D {losses['D'][-1]:+.3f} | G {losses['G'][-1]:+.3f} | {time.time()-t0:.0f}s")
        if ep in (1, 5, 10, 20) or ep == args.epochs:
            G.eval()
            with torch.no_grad(): s = G(fixed).cpu()
            G.train()
            vutils.save_image(s, f"{out}/{name}/epoch_{ep:03d}.png", nrow=8, normalize=True, value_range=(-1, 1), padding=1)
    return G.eval(), losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--label_smooth", action="store_true")
    p.add_argument("--variants", nargs="+", default=["vanilla", "dcgan", "wgan_gp"])
    p.add_argument("--root", default="./data"); p.add_argument("--out", default="outputs_c")
    a = p.parse_args([]); os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])   # -> [-1, 1]
    train = datasets.MNIST(a.root, True, tf, download=True); test = datasets.MNIST(a.root, False, tf, download=True)
    loader = DataLoader(train, a.bs, shuffle=True, drop_last=True, num_workers=2)

    print("Training evaluation classifier ...")
    clf = train_classifier(DataLoader(train, 256, shuffle=True), device)
    real_imgs = torch.stack([test[i][0] for i in range(10000)])
    rf, rp = feats_probs(clf, real_imgs, device)
    print(f"Reference (real test images): IS={inception_score(rp):.3f}")

    results, all_losses = {"real_data_IS": inception_score(rp)}, {}
    for v in a.variants:
        G, losses = train_variant(v, loader, device, a, a.out)
        with torch.no_grad(): fake = torch.cat([G(torch.randn(500, ZDIM, device=device)).cpu() for _ in range(20)])
        ff, fp = feats_probs(clf, fake, device)
        results[v] = {"IS": inception_score(fp), "FID_mnist": fid(rf, ff), "class_balance": fp.argmax(1).tolist().count(0) / len(fp)}
        results[v]["class_hist"] = np.bincount(fp.argmax(1), minlength=10).tolist()   # mode-collapse check
        all_losses[v] = losses; print(v, results[v])
    json.dump({"results": results, "losses": all_losses, "args": vars(a)}, open(f"{a.out}/results.json", "w"), indent=2)

    # loss curves
    fig, ax = plt.subplots(1, len(a.variants), figsize=(5 * len(a.variants), 3.5), squeeze=False)
    for i, v in enumerate(a.variants):
        ax[0, i].plot(all_losses[v]["D"], label="D / critic"); ax[0, i].plot(all_losses[v]["G"], label="G"); ax[0, i].set_title(v); ax[0, i].legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/loss_curves.png", dpi=130); plt.close()
    # side-by-side final samples
    fig, ax = plt.subplots(1, len(a.variants), figsize=(5 * len(a.variants), 5), squeeze=False)
    for i, v in enumerate(a.variants):
        ax[0, i].imshow(plt.imread(f"{a.out}/{v}/epoch_{a.epochs:03d}.png")); ax[0, i].axis("off")
        ax[0, i].set_title(f"{v}\nIS={results[v]['IS']:.2f}  FID={results[v]['FID_mnist']:.1f}")
    plt.tight_layout(); plt.savefig(f"{a.out}/samples_comparison.png", dpi=130)
    print("\nvariant      IS(higher)  FID(lower)")
    for v in a.variants: print(f"{v:<12}{results[v]['IS']:>10.3f}{results[v]['FID_mnist']:>12.2f}")


if __name__ == "__main__":
    main()
