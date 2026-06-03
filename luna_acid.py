import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from src.models import DinoExtractor

N_SPECIAL_TOKENS  = 5
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
INPUT_SIZE        = 224
SHIFT_PX          = 50                                    # applied along both axes
EXPECTED_SHIFT    = float(np.sqrt(2) * SHIFT_PX)         # ≈ 70.7 px diagonal
N_QUAL_SAMPLES    = 4
RNG_SEED          = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShiftResult:
    stem: str
    img_orig: np.ndarray
    img_shifted: np.ndarray
    s_orig: np.ndarray
    s_shifted: np.ndarray
    observed_shift: float          # Euclidean pixel distance between peaks
    equivariance_error: float      # |observed - expected|
    relative_error_pct: float      # error / expected * 100


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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_model(weights: str, adapter_path: str, device: torch.device) -> DinoExtractor:
    extractor = DinoExtractor(weights_path=weights, model_size="vits16")
    extractor.model.load_adapter(adapter_path, adapter_name="pit_adapter")
    extractor.model.set_adapter("pit_adapter")
    return extractor.to(device).eval()


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


def prepare_sample(
    npy_path: Path,
    device: torch.device,
    normalize: transforms.Normalize,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    img = np.load(npy_path).astype(np.float32)
    f_min, f_max = img.min(), img.max()
    img_norm = (img - f_min) / (f_max - f_min + 1e-8)

    tensor = torch.from_numpy(img_norm).unsqueeze(0).repeat(3, 1, 1)
    tensor = TF.resize(tensor, (INPUT_SIZE, INPUT_SIZE))

    fill = tensor.mean().item()
    shifted = TF.affine(tensor, angle=0, translate=(SHIFT_PX, SHIFT_PX),
                        scale=1.0, shear=0, fill=fill)

    x_orig  = normalize(tensor).unsqueeze(0).to(device)
    x_shift = normalize(shifted).unsqueeze(0).to(device)

    img_shifted = shifted.permute(1, 2, 0).numpy()
    return img_norm, img_shifted, x_orig, x_shift


def collect_results(
    paths: list[Path],
    extractor: torch.nn.Module,
    device: torch.device,
) -> list[ShiftResult]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    results = []
    for p in paths:
        img_orig, img_shifted, x_orig, x_shift = prepare_sample(p, device, normalize)
        s_orig    = get_saliency_map(extractor, x_orig)
        s_shifted = get_saliency_map(extractor, x_shift)
        observed  = compute_peak_shift(s_orig, s_shifted)
        error     = abs(observed - EXPECTED_SHIFT)
        results.append(ShiftResult(
            stem                = p.stem,
            img_orig            = img_orig,
            img_shifted         = img_shifted,
            s_orig              = s_orig,
            s_shifted           = s_shifted,
            observed_shift      = observed,
            equivariance_error  = error,
            relative_error_pct  = error / EXPECTED_SHIFT * 100,
        ))
    return results


# ---------------------------------------------------------------------------
# Figure 1 — qualitative shift grid
# ---------------------------------------------------------------------------

def plot_qualitative(results: list[ShiftResult], output_path: str) -> None:
    n = len(results)
    col_titles = [
        "Input (centered)",
        f"Saliency (centered)",
        f"Input (shifted +{SHIFT_PX}px)",
        f"Saliency (shifted)",
    ]

    fig = plt.figure(figsize=(22, 5.5 * n + 0.8))
    gs  = gridspec.GridSpec(
        n + 1, 4,
        height_ratios=[1] * n + [0.045],
        hspace=0.06, wspace=0.06,
        left=0.08, right=0.98, top=0.95, bottom=0.02,
    )
    axes     = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n)])
    cbar_axs = [fig.add_subplot(gs[n, c]) for c in range(4)]

    fig.suptitle(
        f"Translation equivariance test — LoRA DINOv3  "
        f"(expected peak shift: {EXPECTED_SHIFT:.1f} px)",
        fontsize=13, fontweight="bold",
    )

    all_sal  = [r.s_orig for r in results] + [r.s_shifted for r in results]
    sal_vmin = min(a.min() for a in all_sal)
    sal_vmax = max(a.max() for a in all_sal)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=8, fontweight="bold")

    im_sal = None
    for row, res in enumerate(results):
        axes[row, 0].imshow(res.img_orig,    cmap="gray")
        axes[row, 2].imshow(res.img_shifted, cmap="gray")

        im_sal = axes[row, 1].imshow(res.s_orig,    cmap="magma", vmin=sal_vmin, vmax=sal_vmax)
        im_sal = axes[row, 3].imshow(res.s_shifted, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)

        for s_map, ax in [(res.s_orig, axes[row, 1]), (res.s_shifted, axes[row, 3])]:
            pk = np.unravel_index(np.argmax(s_map), s_map.shape)
            ax.plot(pk[1], pk[0], "+", color="cyan", ms=14, mew=2)

        # Equivariance annotation inside the shifted saliency panel
        color = "#2ecc71" if res.relative_error_pct < 30 else "#e74c3c"
        axes[row, 3].text(
            0.03, 0.97,
            f"Observed:  {res.observed_shift:.1f} px\n"
            f"Expected:  {EXPECTED_SHIFT:.1f} px\n"
            f"Error:     {res.equivariance_error:.1f} px  ({res.relative_error_pct:.0f}%)",
            transform=axes[row, 3].transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace", color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.8, alpha=0.92),
            zorder=5,
        )

        axes[row, 0].set_ylabel(res.stem, fontsize=7, labelpad=5)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar under saliency columns only
    for c in [0, 2]:
        cbar_axs[c].set_visible(False)
    for c in [1, 3]:
        cb = fig.colorbar(im_sal, cax=cbar_axs[c], orientation="horizontal")
        cb.set_label("Saliency (shared scale 0–1)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[Line2D([0], [0], marker="+", color="cyan", ms=10, mew=2,
                        linestyle="none", label="Peak activation")],
        loc="lower left", fontsize=9,
        bbox_to_anchor=(0.01, 0.0), framealpha=0.85,
    )

    plt.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — quantitative equivariance error distribution
# ---------------------------------------------------------------------------

def plot_quantitative(results: list[ShiftResult], output_path: str) -> None:
    observed = np.array([r.observed_shift      for r in results])
    errors   = np.array([r.equivariance_error  for r in results])
    rel_errs = np.array([r.relative_error_pct  for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    fig.suptitle(
        f"Translation equivariance — LoRA DINOv3  "
        f"(n={len(results)}, expected shift {EXPECTED_SHIFT:.1f} px)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    rng = np.random.default_rng(RNG_SEED)

    def _strip_hist(ax, values, xlabel, color, vline=None, vline_label=None):
        ax.hist(values, bins=20, color=color, alpha=0.45, edgecolor="white", linewidth=0.5)
        jitter = rng.uniform(-0.3, 0.3, len(values))
        y_strip = np.full(len(values), ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 0.5)
        # draw strip at bottom after hist is done
        ax.scatter(values, np.zeros(len(values)) - 0.5, color=color, s=18, alpha=0.7,
                   clip_on=False, zorder=3)
        if vline is not None:
            ax.axvline(vline, color="#c0392b", linestyle="--", linewidth=1.6, label=vline_label)
        median = float(np.median(values))
        ax.axvline(median, color=color, linestyle="-", linewidth=2.0, label=f"Median {median:.1f}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.9)

    _strip_hist(
        axes[0], observed,
        xlabel=f"Observed peak shift (px)",
        color="#2980b9",
        vline=EXPECTED_SHIFT,
        vline_label=f"Expected {EXPECTED_SHIFT:.1f} px",
    )

    _strip_hist(
        axes[1], errors,
        xlabel="Absolute equivariance error (px)",
        color="#8e44ad",
    )

    _strip_hist(
        axes[2], rel_errs,
        xlabel="Relative equivariance error (%)",
        color="#16a085",
    )

    # Summary stats box
    summary = (
        f"n = {len(results)}\n"
        f"Mean error:   {errors.mean():.1f} px\n"
        f"Median error: {np.median(errors):.1f} px\n"
        f"σ error:      {errors.std():.1f} px\n"
        f"<30% error:   {(rel_errs < 30).mean()*100:.0f}% of samples"
    )
    fig.text(
        0.99, 0.98, summary,
        ha="right", va="top", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", lw=1),
    )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_acid_test() -> None:
    device       = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    weights      = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth"
    adapter_path = "models/best_lora_pit_model/best_lora_pit_model_qkv_proj_fc1_fc2"

    extractor = load_model(weights, adapter_path, device)
    print(f"Adapter weight sum: "
          f"{extractor.model.base_model.model.blocks[0].attn.qkv.lora_A.pit_adapter.weight.sum().item():.8f}")

    all_pits = list(Path("data/processed/dataset/test/pits").glob("*.npy"))
    if not all_pits:
        print("No test pits found.")
        return

    print(f"Running equivariance test on {len(all_pits)} pits ...")
    all_results = collect_results(all_pits, extractor, device)

    qual = {r.stem for r in sorted(all_results, key=lambda r: r.equivariance_error)[:N_QUAL_SAMPLES]}

    stem = lambda s: build_output_name(adapter_path, s)

    plot_qualitative(
        [r for r in all_results if r.stem in qual],
        stem("luna_fig_equivariance_qual"),
    )
    plot_quantitative(all_results, stem("luna_fig_equivariance_quant"))


if __name__ == "__main__":
    run_acid_test()