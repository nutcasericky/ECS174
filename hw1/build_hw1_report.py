"""Build report figures, a results summary, and a PDF from saved HW1 measurements."""

import csv
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "report_figures"
RUNS = ["baseline", "linear", "mlp", "mlp_no_relu", "batch_1", "batch_1000",
        "lr_10", "lr_0_1", "lr_0_01", "augmentation", "mse"]


def load_histories():
    histories = {}
    for name in RUNS:
        with (RESULTS / name / "metrics.csv").open() as handle:
            history = [{key: float(value) for key, value in row.items()}
                       for row in csv.DictReader(handle)]
        assert history and [row["epoch"] for row in history] == list(range(1, len(history) + 1))
        if name != "lr_10":
            assert all(math.isfinite(row[key]) for row in history for key in
                       ["train_loss", "test_loss", "train_accuracy", "test_accuracy"])
        histories[name] = history
    return histories


def curves(series, path, title, single=False, failure_note=None, loss_name="Training loss"):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for label, history in series:
        epochs = [row["epoch"] for row in history]
        marker = "." if len(history) <= 10 else None
        for ax, key in zip(axes, ["train_loss", "train_accuracy", "test_accuracy"]):
            ax.plot(epochs, [row[key] for row in history], label=label,
                    linewidth=1.6, marker=marker)
        if single:
            axes[0].plot(epochs, [row["test_loss"] for row in history],
                         label="Test loss", linestyle="--", linewidth=1.6, marker=marker)
    for ax, subtitle in zip(axes, [loss_name, "Training accuracy", "Test accuracy"]):
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(fontsize=8, loc="best")
    for ax in axes[1:]:
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 100)
    if failure_note:
        fig.text(0.5, 0.015, failure_note, color="#a32323", fontsize=9, ha="center")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.05 if failure_note else 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures(histories):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, filename, title in [
        ("linear", "linear_50.png", "Single fully connected layer: 50 epochs"),
        ("mlp_no_relu", "no_relu_50.png", "MLP without ReLU: 50 epochs"),
        ("baseline", "cnn_50.png", "Tutorial CNN: 50 epochs"),
    ]:
        curves([("Training loss", histories[name])], FIGURES / filename,
               title, single=True, loss_name="Cross-entropy loss")
    curves([("With ReLU", histories["mlp"][:5]),
            ("Without ReLU", histories["mlp_no_relu"][:5])],
           FIGURES / "relu_comparison.png", "ReLU comparison: first 5 epochs")
    curves([("Batch 1", histories["batch_1"][:5]),
            ("Batch 4", histories["baseline"][:5]),
            ("Batch 1000", histories["batch_1000"][:5])],
           FIGURES / "batch_comparison.png", "Batch size comparison: first 5 epochs")
    curves([("LR 0.001", histories["baseline"][:5]),
            ("LR 0.01", histories["lr_0_01"][:5]),
            ("LR 0.1", histories["lr_0_1"][:5])],
           FIGURES / "lr_comparison.png", "Learning rate comparison: first 5 epochs",
           failure_note="LR 10 failed during epoch 1; no finite per-epoch measurements exist.")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.4))
    for ax, title in zip(axes, ["Loss", "Training accuracy", "Test accuracy"]):
        ax.set_title(title, fontsize=11)
        ax.text(0.5, 0.5, "Unavailable\nNon-finite loss before\nfirst epoch evaluation",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#a32323")
        ax.set(xticks=[], yticks=[])
    fig.suptitle("Learning rate 10: numerical divergence during epoch 1", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "lr_10_failed.png", dpi=200)
    plt.close(fig)
    curves([("No augmentation", histories["baseline"][:5]),
            ("Augmented training", histories["augmentation"][:5])],
           FIGURES / "augmentation_comparison.png", "Augmentation: first 5 epochs")
    curves([("Training loss", histories["baseline"][:5])], FIGURES / "ce_5.png",
           "Cross entropy: first 5 epochs", single=True, loss_name="Cross-entropy loss")
    curves([("Training loss", histories["mse"][:5])], FIGURES / "mse_5.png",
           "One-hot MSE on raw output scores: 5 epochs", single=True,
           loss_name="Mean squared error")


def write_summary(histories):
    fields = ["experiment", "epochs_completed", "status", "parameters", "train_loss",
              "test_loss", "train_accuracy", "test_accuracy", "mean_epoch_seconds"]
    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, history in histories.items():
            final = history[-1]
            status = "complete"
            metadata_path = RESULTS / name / "run.json"
            if metadata_path.exists():
                status = json.loads(metadata_path.read_text())["status"]
            parameters = 30730 if name == "linear" else 346994 if name.startswith("mlp") else 62006
            writer.writerow(dict(
                experiment=name, epochs_completed=0 if status == "diverged" else len(history),
                status=status, parameters=parameters,
                **{key: final[key] for key in fields[4:8]},
                mean_epoch_seconds=sum(row["seconds"] for row in history) / len(history),
            ))


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.replace("**", "").replace("`", "")


class ReportPDF:
    """Lay out the Markdown report using the already installed Matplotlib stack."""

    def __init__(self, path):
        self.pdf = PdfPages(path, metadata={
            "Title": "ECS 174 — Homework 1", "Subject": "CIFAR-10 experiment report",
        })
        self.figure = None
        self.pages = 0
        self.left, self.right = 43, 569
        self.width = self.right - self.left
        self.measure = TextToPath()
        self.new_page()

    def new_page(self):
        if self.figure is not None:
            self.pdf.savefig(self.figure)
            plt.close(self.figure)
        self.pages += 1
        self.figure = plt.figure(figsize=(8.5, 11))
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 612), ylim=(0, 792))
        self.ax.axis("off")
        self.ax.text(self.left, 768, "ECS 174  |  Homework 1", fontsize=8.5,
                     color="#526071", va="top")
        self.ax.plot([self.left, self.right], [753, 753], color="#c5cdd7", linewidth=0.6)
        self.ax.text(self.right, 24, str(self.pages), fontsize=8.5, color="#526071", ha="right")
        self.y = 738

    def ensure(self, height):
        if self.y - height < 44:
            self.new_page()

    def wrap(self, text, width, size=10.2, bold=False):
        prop = FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")
        lines, current = [], ""
        for word in plain(text).split():
            candidate = f"{current} {word}".strip()
            measured = self.measure.get_text_width_height_descent(candidate, prop, ismath=False)[0]
            if current and measured > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [""]

    def paragraph(self, text, indent=0):
        lines = self.wrap(text, self.width - indent)
        for index, line in enumerate(lines):
            self.ensure(28 if index == 0 and len(lines) > 1 else 14)
            self.ax.text(self.left + indent, self.y, line, fontsize=10.2, va="top",
                         fontfamily="DejaVu Sans", color="#18212b")
            self.y -= 14
        self.y -= 6

    def heading(self, text, level):
        size = 21 if level == 1 else 14
        lines = self.wrap(text, self.width, size=size, bold=True)
        self.ensure(len(lines) * (size + 4) + 70)
        if level > 1:
            self.y -= 8
        for line in lines:
            self.ax.text(self.left, self.y, line, fontsize=size, va="top", weight="bold",
                         fontfamily="DejaVu Sans", color="#15314c")
            self.y -= size + 4
        self.y -= 7

    def table(self, rows):
        count = len(rows[0])
        if count == 2:
            widths = [self.width * 0.64, self.width * 0.36]
        else:
            widths = [self.width * 0.32] + [self.width * 0.68 / (count - 1)] * (count - 1)
        wrapped = [[self.wrap(cell, width - 12, size=9, bold=(i == 0))
                    for cell, width in zip(row, widths)] for i, row in enumerate(rows)]
        heights = [max(len(cell) for cell in row) * 12 + 12 for row in wrapped]
        self.ensure(sum(heights) + 12)
        for i, (row, height) in enumerate(zip(wrapped, heights)):
            background = "#e6eef6" if i == 0 else "#f5f7fa" if i % 2 else "white"
            self.ax.add_patch(Rectangle((self.left, self.y - height), self.width, height,
                                        facecolor=background, edgecolor="#d3dbe5", linewidth=0.5))
            x = self.left
            for cell, width in zip(row, widths):
                for j, line in enumerate(cell):
                    self.ax.text(x + 6, self.y - 6 - j * 12, line, fontsize=9, va="top",
                                 weight="bold" if i == 0 else "normal", fontfamily="DejaVu Sans")
                x += width
            self.y -= height
        self.y -= 13

    def image(self, caption, path):
        pixels = mpimg.imread(path)
        height = self.width * pixels.shape[0] / pixels.shape[1]
        lines = self.wrap(caption, self.width, size=8.5)
        self.ensure(height + len(lines) * 11 + 16)
        image_ax = self.figure.add_axes([
            self.left / 612, (self.y - height) / 792, self.width / 612, height / 792,
        ], zorder=-1)
        image_ax.imshow(pixels)
        image_ax.axis("off")
        self.y -= height + 4
        for line in lines:
            self.ax.text(self.left, self.y, line, fontsize=8.5, color="#526071", va="top")
            self.y -= 11
        self.y -= 12

    def close(self):
        self.pdf.savefig(self.figure)
        plt.close(self.figure)
        self.pdf.close()


def export_pdf(markdown_path, pdf_path):
    lines = markdown_path.read_text().splitlines()
    report = ReportPDF(pdf_path)
    i = 0
    try:
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith("#"):
                report.heading(line.lstrip("# "), len(line) - len(line.lstrip("#")))
                i += 1
            elif line.startswith("!["):
                match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
                assert match, line
                report.image(match.group(1), ROOT / match.group(2))
                i += 1
            elif line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                        rows.append(cells)
                    i += 1
                report.table(rows)
            else:
                paragraph = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "![", "- ")):
                    if re.match(r"\d+\. ", lines[i]):
                        break
                    paragraph.append(lines[i].strip())
                    i += 1
                report.paragraph(" ".join(paragraph))
    finally:
        report.close()
    return report.pages


def main():
    histories = load_histories()
    make_figures(histories)
    write_summary(histories)
    pdf_path = ROOT / "HW1_report.pdf"
    pages = export_pdf(ROOT / "HW1_report.md", pdf_path)
    print(f"Created {pdf_path.name}: {pages} pages")
    print(f"Created {len(list(FIGURES.glob('*.png')))} report figures and results/summary.csv")


if __name__ == "__main__":
    main()
