import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json, os
from muon import Muon

device = "cuda" if torch.cuda.is_available() else "cpu"

mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
train_loader = DataLoader(torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)])), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

def make_cnn():
    return nn.Sequential(
        nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128*8*8, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 10)
    ).to(device)

loss_fn = nn.CrossEntropyLoss()

def train(model, opt, loader):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        l = loss_fn(model(x), y)
        l.backward()
        opt.step()
        total += l.item() * len(x)
    return total / len(loader.dataset)

def val(model, loader):
    model.eval()
    loss, ok = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += loss_fn(out, y).item() * len(x)
            ok += out.argmax(1).eq(y).sum().item()
    return loss / len(loader.dataset), ok / len(loader.dataset)

# grid search по lr
best_lr_sgd, best_acc_sgd = 0.01, -1
for lr in [0.1, 0.05, 0.01]:
    torch.manual_seed(0)
    m = make_cnn()
    o = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    for _ in range(3):
        train(m, o, train_loader)
    _, acc = val(m, val_loader)
    print(f"sgd lr={lr} acc={acc:.4f}")
    if acc > best_acc_sgd:
        best_acc_sgd, best_lr_sgd = acc, lr

best_lr_adam, best_acc_adam = 3e-4, -1
for lr in [1e-3, 3e-4, 1e-4]:
    torch.manual_seed(0)
    m = make_cnn()
    o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(3):
        train(m, o, train_loader)
    _, acc = val(m, val_loader)
    print(f"adam lr={lr} acc={acc:.4f}")
    if acc > best_acc_adam:
        best_acc_adam, best_lr_adam = acc, lr

best_lr_muon, best_acc_muon = 0.01, -1
for lr in [0.05, 0.02, 0.01]:
    torch.manual_seed(0)
    m = make_cnn()
    o = Muon([p for p in m.parameters() if p.ndim >= 2], lr=lr, adamw_params=[p for p in m.parameters() if p.ndim < 2], adamw_lr=3e-4, adamw_wd=5e-4)
    for _ in range(3):
        train(m, o, train_loader)
    _, acc = val(m, val_loader)
    print(f"muon lr={lr} acc={acc:.4f}")
    if acc > best_acc_muon:
        best_acc_muon, best_lr_muon = acc, lr

print(f"best lrs: sgd={best_lr_sgd} adam={best_lr_adam} muon={best_lr_muon}")

# полные прогоны
torch.manual_seed(42)
sgd_model = make_cnn()
sgd_opt = torch.optim.SGD(sgd_model.parameters(), lr=best_lr_sgd, momentum=0.9, nesterov=True, weight_decay=5e-4)

torch.manual_seed(42)
adam_model = make_cnn()
adam_opt = torch.optim.Adam(adam_model.parameters(), lr=best_lr_adam, weight_decay=5e-4)

torch.manual_seed(42)
muon_model = make_cnn()
muon_opt = Muon([p for p in muon_model.parameters() if p.ndim >= 2], lr=best_lr_muon, adamw_params=[p for p in muon_model.parameters() if p.ndim < 2], adamw_lr=3e-4, adamw_wd=5e-4)

sgd_tl, sgd_vl, sgd_va = [], [], []
adam_tl, adam_vl, adam_va = [], [], []
muon_tl, muon_vl, muon_va = [], [], []

for ep in range(1, 16):
    sgd_tl.append(train(sgd_model, sgd_opt, train_loader))
    vl, va = val(sgd_model, val_loader)
    sgd_vl.append(vl)
    sgd_va.append(va)

    adam_tl.append(train(adam_model, adam_opt, train_loader))
    vl, va = val(adam_model, val_loader)
    adam_vl.append(vl)
    adam_va.append(va)

    muon_tl.append(train(muon_model, muon_opt, train_loader))
    vl, va = val(muon_model, val_loader)
    muon_vl.append(vl)
    muon_va.append(va)

    print(f"ep{ep:2d} | sgd={sgd_va[-1]:.4f} adam={adam_va[-1]:.4f} muon={muon_va[-1]:.4f}")

os.makedirs("./results", exist_ok=True)
torch.save(sgd_model.state_dict(), "./results/cifar10_model_SGD.pt")
torch.save(adam_model.state_dict(), "./results/cifar10_model_Adam.pt")
torch.save(muon_model.state_dict(), "./results/cifar10_model_Muon_AdamW.pt")

results = {
    "SGD": {"best_lr": best_lr_sgd, "train_losses": sgd_tl, "val_losses": sgd_vl, "val_accs": sgd_va, "final_val_acc": sgd_va[-1]},
    "Adam": {"best_lr": best_lr_adam, "train_losses": adam_tl, "val_losses": adam_vl, "val_accs": adam_va, "final_val_acc": adam_va[-1]},
    "Muon+AdamW": {"best_lr": best_lr_muon, "train_losses": muon_tl, "val_losses": muon_vl, "val_accs": muon_va, "final_val_acc": muon_va[-1]},
}
with open("./results/cifar10_results.json", "w") as f:
    json.dump(results, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("CNN на CIFAR-10: сравнение оптимизаторов", fontweight="bold")

axes[0].plot(sgd_tl, color="#2196F3", label=f"SGD (lr={best_lr_sgd})")
axes[0].plot(adam_tl, color="#FF9800", label=f"Adam (lr={best_lr_adam})")
axes[0].plot(muon_tl, color="#4CAF50", label=f"Muon+AdamW (lr={best_lr_muon})")

axes[1].plot(sgd_vl, color="#2196F3", label=f"SGD (lr={best_lr_sgd})")
axes[1].plot(adam_vl, color="#FF9800", label=f"Adam (lr={best_lr_adam})")
axes[1].plot(muon_vl, color="#4CAF50", label=f"Muon+AdamW (lr={best_lr_muon})")

axes[2].plot([a*100 for a in sgd_va], color="#2196F3", label=f"SGD (lr={best_lr_sgd})")
axes[2].plot([a*100 for a in adam_va], color="#FF9800", label=f"Adam (lr={best_lr_adam})")
axes[2].plot([a*100 for a in muon_va], color="#4CAF50", label=f"Muon+AdamW (lr={best_lr_muon})")

for ax, t, yl in zip(axes, ["Train Loss", "Val Loss", "Val Accuracy (%)"], ["Loss", "Loss", "Accuracy (%)"]):
    ax.set(title=t, xlabel="Epoch", ylabel=yl)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./results/cifar10_curves.png", dpi=150, bbox_inches="tight")