"""HW1 Part 1: the tutorial CNN, with measurements saved after every epoch."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        if step % 2000 == 0:
            print(f"  Training batch {step}/{len(loader)} | batch loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        loss_sum += criterion(scores, labels) * labels.size(0)
        correct += (scores.argmax(dim=1) == labels).sum()
        total += labels.size(0)
    return loss_sum.item() / total, 100.0 * correct.item() / total


def save_curves(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="Test")
    axes[0].set(title="Cross-entropy loss", ylabel="Loss")
    axes[0].legend()
    for ax, key, title in [
        (axes[1], "train_accuracy", "Training accuracy"),
        (axes[2], "test_accuracy", "Test accuracy"),
    ]:
        ax.plot(epochs, [row[key] for row in history])
        ax.set(title=title, ylabel="Accuracy (%)", ylim=(0, 100))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")

    torch.manual_seed(0)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)

    # Zero workers keeps the first version simple and avoids macOS spawn issues.
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # Larger evaluation batches are faster and do not change training updates.
    train_eval_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | Parameters: {parameters:,}", flush=True)
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}", flush=True)
    print("Each epoch trains, then evaluates all training and test images.", flush=True)

    history = []
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy", "seconds",
        ])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = perf_counter()
            train_one_epoch(model, trainloader, criterion, optimizer, device)
            train_loss, train_accuracy = evaluate(model, train_eval_loader, criterion, device)
            test_loss, test_accuracy = evaluate(model, testloader, criterion, device)
            row = dict(
                epoch=epoch, train_loss=train_loss, test_loss=test_loss,
                train_accuracy=train_accuracy, test_accuracy=test_accuracy,
                seconds=perf_counter() - started,
            )
            history.append(row)
            writer.writerow(row)
            handle.flush()
            save_curves(history, args.output / "curves.png")
            print(
                f"Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | "
                f"train {train_accuracy:.2f}% | test {test_accuracy:.2f}% | "
                f"{row['seconds']:.1f}s", flush=True,
            )

    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)},
               args.output / "model.pt")
    print(f"Saved metrics.csv, curves.png, and model.pt in {args.output}", flush=True)


if __name__ == "__main__":
    main()
