import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path
from dataclasses import dataclass
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from src.models import DinoExtractor

N_SPECIAL_TOKENS = 5
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
INPUT_SIZE       = 224
N_QUAL_SAMPLES   = 4
TOPK_FRACTION    = 0.20
RNG_SEED         = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SaliencyResult:
    label: str
    stem: str
    img_norm: np.ndarray
    s_base: np.ndarray
    s_lora: np.ndarray
    diff: np.ndarray
    peak_dist: float
    topk_iou: float
    mad: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def get_saliency_map(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    features = {}

    def hook(module, input, output):
        features["val"] = output[0] if isinstance(output, (list, tuple)) else output

    handle = model.model.model.blocks[-1].register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    feat = features["val"][0, N_SPECIAL_TOKENS:, :]
    saliency = torch.norm(feat, dim=-1)
    grid_size = int(np.sqrt(feat.shape[0]))
    saliency = saliency.reshape(grid_size, grid_size)
    saliency = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    return saliency.cpu().numpy()


def compute_peak_shift(s1: np.ndarray, s2: np.ndarray) -> float:
    p1 = np.array(np.unravel_index(np.argmax(s1), s1.shape), dtype=float)
    p2 = np.array(np.unravel_index(np.argmax(s2), s2.shape), dtype=float)
    return float(np.linalg.norm(p1 - p2))


def compute_topk_iou(s1: np.ndarray, s2: np.ndarray, k: float = TOPK_FRACTION) -> float:
    thr1, thr2 = np.percentile(s1, (1 - k) * 100), np.percentile(s2, (1 - k) * 100)
    m1, m2 = s1 >= thr1, s2 >= thr2
    union = (m1 | m2).sum()
    return float((m1 & m2).sum() / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_model(weights: str, device: torch.device) -> DinoExtractor:
    return DinoExtractor(weights_path=weights, model_size="vits16").to(device).eval()


def load_input(path: Path, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    img = np.load(path).astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    tensor = torch.from_numpy(img_norm).unsqueeze(0).repeat(3, 1, 1)
    tensor = TF.resize(tensor, (INPUT_SIZE, INPUT_SIZE))
    x = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    return img_norm, x.unsqueeze(0).to(device)


def build_output_name(adapter_path: str, stem: str) -> str:
    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    peft_type = cfg.get("peft_type", "adapter").lower()
    r         = cfg.get("r", "?")
    alpha     = cfg.get("lora_alpha", "?")
    modules   = "-".join(cfg.get("target_modules", []))
    dora      = "_dora" if cfg.get("use_dora") else ""
    rslora    = "_rs"   if cfg.get("use_rslora") else ""
    return f"{stem}_{peft_type}_r{r}a{alpha}_{modules}{dora}{rslora}.svg"


def collect_results(
    paths: list[Path],
    label: str,
    model_base: torch.nn.Module,
    model_lora: torch.nn.Module,
    device: torch.device,
) -> list[SaliencyResult]:
    results = []
    for p in paths:
        img_norm, x = load_input(p, device)
        s_base = get_saliency_map(model_base, x)
        s_lora = get_saliency_map(model_lora, x)
        diff   = s_lora - s_base
        results.append(SaliencyResult(
            label     = label,
            stem      = p.stem,
            img_norm  = img_norm,
            s_base    = s_base,
            s_lora    = s_lora,
            diff      = diff,
            peak_dist = compute_peak_shift(s_base, s_lora),
            topk_iou  = compute_topk_iou(s_base, s_lora),
            mad       = float(np.abs(diff).mean()),
        ))
    return results


# ---------------------------------------------------------------------------
# Figure 1 & 2 — qualitative grids
# ---------------------------------------------------------------------------

def _plot_qualitative_grid(
    results: list[SaliencyResult],
    col_titles: list[str],
    fig_title: str,
    output_path: str,
) -> None:
    n = len(results)

    # GridSpec: 4 image rows + 1 thin colorbar row, 4 columns
    fig = plt.figure(figsize=(22, 5.5 * n + 0.8))
    gs = gridspec.GridSpec(
        n + 1, 4,
        height_ratios=[1] * n + [0.045],
        hspace=0.06,
        wspace=0.06,
        left=0.08, right=0.98, top=0.96, bottom=0.02,
    )

    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n)])
    cbar_axes = [fig.add_subplot(gs[n, c]) for c in range(4)]

    fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    all_sal = [r.s_base for r in results] + [r.s_lora for r in results]
    sal_vmin     = min(a.min() for a in all_sal)
    sal_vmax     = max(a.max() for a in all_sal)
    diff_abs_max = max(np.abs(r.diff).max() for r in results)
    diff_norm    = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=8, fontweight="bold")

    im_sal = im_diff = None
    for row, res in enumerate(results):
        axes[row, 0].imshow(res.img_norm, cmap="gray")

        im_sal = axes[row, 1].imshow(res.s_base, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)
        pk = np.unravel_index(np.argmax(res.s_base), res.s_base.shape)
        axes[row, 1].plot(pk[1], pk[0], "+", color="cyan", ms=14, mew=2)

        axes[row, 2].imshow(res.s_lora, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)
        pk = np.unravel_index(np.argmax(res.s_lora), res.s_lora.shape)
        axes[row, 2].plot(pk[1], pk[0], "+", color="cyan", ms=14, mew=2)

        im_diff = axes[row, 3].imshow(res.diff, cmap="RdBu_r", norm=diff_norm)

        axes[row, 0].set_ylabel(res.stem, fontsize=7, labelpad=5)

        # Metrics as a clean annotation inside the diff panel (top-left, white bg)
        axes[row, 3].text(
            0.03, 0.97,
            f"Δpeak {res.peak_dist:.0f}px\nIoU@20% {res.topk_iou:.2f}\nMAD {res.mad:.3f}",
            transform=axes[row, 3].transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.82),
        )

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbars in dedicated bottom row
    # Columns 0 and 1-2: hide (gray labels instead)
    for c in range(3):
        cbar_axes[c].set_visible(False)

    # Saliency label under cols 1-2
    fig.text(
        (gs.get_grid_positions(fig)[3][1] + gs.get_grid_positions(fig)[3][2]) / 2,
        0.005,
        "Saliency (normalised, shared scale 0–1)",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )

    # Diff colorbar under col 3 only
    cb = fig.colorbar(im_diff, cax=cbar_axes[3], orientation="horizontal")
    cb.set_label("LoRA − Base", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)

    # Peak activation legend
    legend = [Line2D([0], [0], marker="+", color="cyan", ms=10, mew=2,
                     linestyle="none", label="Peak activation")]
    fig.legend(handles=legend, loc="lower left", fontsize=9,
               bbox_to_anchor=(0.01, 0.0), framealpha=0.85)

    plt.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_qualitative_pits(results: list[SaliencyResult], output_path: str) -> None:
    _plot_qualitative_grid(
        results,
        col_titles=["NAC Image", "Base DINOv3 (zero-shot)", "LoRA DINOv3 (fine-tuned)", "Difference (LoRA − Base)"],
        fig_title="Patch-feature saliency comparison — lunar pit samples",
        output_path=output_path,
    )


def plot_qualitative_nonpits(results: list[SaliencyResult], output_path: str) -> None:
    _plot_qualitative_grid(
        results,
        col_titles=["NAC Image", "Base DINOv3 (zero-shot)", "LoRA DINOv3 (fine-tuned)", "Difference (LoRA − Base)"],
        fig_title="Patch-feature saliency comparison — non-pit control samples",
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Figure 3 — quantitative metrics
# ---------------------------------------------------------------------------

def plot_quantitative(
    pit_results: list[SaliencyResult],
    nonpit_results: list[SaliencyResult],
    output_path: str,
) -> None:
    metrics = [
        ("peak_dist", "Peak shift (px)",        r"Euclidean dist. between $\arg\max$ activations"),
        ("topk_iou",  f"Top-{int(TOPK_FRACTION*100)}% IoU", "IoU of top-20% activated patches"),
        ("mad",       "Mean abs. difference",   "MAD of normalised saliency maps"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Quantitative saliency shift: LoRA fine-tuning vs. zero-shot baseline",
        fontsize=12, fontweight="bold", y=1.02,
    )

    colors    = {"pit": "#c0392b", "non-pit": "#2980b9"}
    positions = {"pit": 1,         "non-pit": 2}
    rng       = np.random.default_rng(RNG_SEED)

    for ax, (attr, metric_label, xlabel) in zip(axes, metrics):
        for label, res_list in [("pit", pit_results), ("non-pit", nonpit_results)]:
            values = np.array([getattr(r, attr) for r in res_list])
            pos    = positions[label]
            color  = colors[label]

            vp = ax.violinplot(values, positions=[pos], widths=0.5,
                               showmedians=False, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.45)
            for part in ("cmins", "cmaxes", "cbars"):
                vp[part].set_color(color)
                vp[part].set_linewidth(1.4)

            # Jittered strip
            jitter = rng.uniform(-0.11, 0.11, len(values))
            ax.scatter(
                [pos + j for j in jitter], values,
                color=color, s=22, alpha=0.75, zorder=3,
            )

            # Median line drawn manually so we control its z-order and style
            median = float(np.median(values))
            ax.hlines(median, pos - 0.22, pos + 0.22,
                      color=color, linewidth=2.5, zorder=4)

            # Median label above the violin body, white background
            y_max = values.max()
            ax.text(
                pos, y_max,
                f"med={median:.2f}",
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8, alpha=0.95),
                zorder=5,
            )

        ax.set_title(metric_label, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(xlabel, fontsize=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pit", "Non-pit"], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors["pit"],     alpha=0.7, label="Pit"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["non-pit"], alpha=0.7, label="Non-pit"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.07), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> None:
    device       = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    weights      = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth"
    adapter_path = "models/lunar_dinov3_lora/standard_lora"

    model_base = load_model(weights, device)
    model_lora = load_model(weights, device)
    model_lora.model.load_adapter(adapter_path, adapter_name="pit_fine_tuned")
    model_lora.model.set_adapter("pit_fine_tuned")

    rng = np.random.default_rng(RNG_SEED)

    all_pits    = list(Path("data/processed/dataset/test/pits").glob("*.npy"))
    all_nonpits = list(Path("data/processed/dataset/test/negatives").glob("*.npy"))

    qual_pits    = set(p.stem for p in rng.choice(all_pits,    size=min(N_QUAL_SAMPLES, len(all_pits)),    replace=False))
    qual_nonpits = set(p.stem for p in rng.choice(all_nonpits, size=min(N_QUAL_SAMPLES, len(all_nonpits)), replace=False))

    print(f"Computing saliency for {len(all_pits)} pits and {len(all_nonpits)} non-pits ...")
    pit_results    = collect_results(all_pits,    "pit",     model_base, model_lora, device)
    nonpit_results = collect_results(all_nonpits, "non-pit", model_base, model_lora, device)

    stem = lambda s: build_output_name(adapter_path, s)

    plot_qualitative_pits(
        [r for r in pit_results    if r.stem in qual_pits],
        stem("luna_fig1_pits"),
    )
    plot_qualitative_nonpits(
        [r for r in nonpit_results if r.stem in qual_nonpits],
        stem("luna_fig2_nonpits"),
    )
    plot_quantitative(pit_results, nonpit_results, stem("luna_fig3_quantitative"))


if __name__ == "__main__":
    run_all()