import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from dataclasses import dataclass, field
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from src.models import DinoExtractor

# ---------------------------------------------------------------------------
# Configuration — edit here to switch models
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    label: str            # display name in figure
    weights: str          # path to base weights
    adapter_path: str     # path to adapter directory
    adapter_name: str     # internal PEFT adapter name

MODELS: list[ModelConfig] = [
    ModelConfig(
        label        = "Standard LoRA",
        weights      = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth",
        adapter_path = "models/best_lora_pit_model/best_lora_pit_model_qkv_proj_fc1_fc2",
        adapter_name = "standard_lora",
    ),
    ModelConfig(
        label        = "Matryoshka LoRA",
        weights      = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth",
        adapter_path = "models/best_lora_pit_model/best_mat_lora_pit_model_qkv_proj_fc1_fc2",
        adapter_name = "matryoshka_lora",
    ),
]

N_SAMPLES       = 4
SHIFT_PX        = 50
EXPECTED_SHIFT  = float(np.sqrt(2) * SHIFT_PX)
N_SPECIAL_TOKENS = 5
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
INPUT_SIZE      = 224
RNG_SEED        = 67

# Fraction of near-zero pixels to flag as artifact sample
ARTIFACT_BLACK_THRESHOLD  = 0.05   # pixel value
ARTIFACT_BLACK_RATIO_MIN  = 0.10   # at least 10% black pixels


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SaliencyPair:
    stem: str
    img_norm: np.ndarray
    img_shifted: np.ndarray
    is_artifact: bool
    per_model: dict[str, tuple[np.ndarray, np.ndarray]]  # label → (s_centered, s_shifted)
    per_model_error: dict[str, float]                     # label → equivariance_error


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


def is_artifact(img_norm: np.ndarray) -> bool:
    return (img_norm < ARTIFACT_BLACK_THRESHOLD).mean() > ARTIFACT_BLACK_RATIO_MIN


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_model(cfg: ModelConfig, device: torch.device) -> DinoExtractor:
    extractor = DinoExtractor(weights_path=cfg.weights, model_size="vits16")
    extractor.model.load_adapter(cfg.adapter_path, adapter_name=cfg.adapter_name)
    extractor.model.set_adapter(cfg.adapter_name)
    return extractor.to(device).eval()


def build_output_name(configs: list[ModelConfig], stem: str) -> str:
    # derive tag from first model's adapter config
    config_path = Path(configs[0].adapter_path) / "adapter_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    peft_type = cfg.get("peft_type", "adapter").lower()
    r         = cfg.get("r", "?")
    alpha     = cfg.get("lora_alpha", "?")
    modules   = "-".join(cfg.get("target_modules", []))
    return f"{stem}_{peft_type}_r{r}a{alpha}_{modules}.svg"


def prepare_sample(
    npy_path: Path,
    device: torch.device,
    normalize: transforms.Normalize,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    img = np.load(npy_path).astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    tensor = torch.from_numpy(img_norm).unsqueeze(0).repeat(3, 1, 1)
    tensor = TF.resize(tensor, (INPUT_SIZE, INPUT_SIZE))

    fill    = tensor.mean().item()
    shifted = TF.affine(tensor, angle=0, translate=(SHIFT_PX, SHIFT_PX),
                        scale=1.0, shear=0, fill=fill)

    x_orig  = normalize(tensor).unsqueeze(0).to(device)
    x_shift = normalize(shifted).unsqueeze(0).to(device)
    return img_norm, shifted.permute(1, 2, 0).numpy(), x_orig, x_shift


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_pairs(
    paths: list[Path],
    models: dict[str, torch.nn.Module],
    device: torch.device,
) -> list[SaliencyPair]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    pairs = []
    for p in paths:
        img_norm, img_shifted, x_orig, x_shift = prepare_sample(p, device, normalize)
        per_model       = {}
        per_model_error = {}
        for label, model in models.items():
            s_c = get_saliency_map(model, x_orig)
            s_s = get_saliency_map(model, x_shift)
            observed = compute_peak_shift(s_c, s_s)
            per_model[label]       = (s_c, s_s)
            per_model_error[label] = abs(observed - EXPECTED_SHIFT)
        pairs.append(SaliencyPair(
            stem            = p.stem,
            img_norm        = img_norm,
            img_shifted     = img_shifted,
            is_artifact     = is_artifact(img_norm),
            per_model       = per_model,
            per_model_error = per_model_error,
        ))
    return pairs


def select_samples(all_pairs: list[SaliencyPair], n: int) -> list[SaliencyPair]:
    # Artifact sample always first (if present), rest = lowest mean equivariance error
    artifacts  = [p for p in all_pairs if p.is_artifact]
    non_art    = [p for p in all_pairs if not p.is_artifact]
    non_art    = sorted(non_art, key=lambda p: np.mean(list(p.per_model_error.values())))
    selected   = (artifacts[:1] + non_art)[:n]
    return selected


# ---------------------------------------------------------------------------
# Figure — 4 rows × 6 columns with visual model separator
# ---------------------------------------------------------------------------

def plot_comparison(
    samples: list[SaliencyPair],
    model_labels: list[str],
    output_path: str,
) -> None:
    n      = len(samples)
    label_a, label_b = model_labels

    # GridSpec: [raw | sal_A_c | sal_A_s | spacer | sal_B_c | sal_B_s]
    # + 1 thin colorbar row at bottom
    col_widths = [1, 1, 1, 0.06, 1, 1]
    fig = plt.figure(figsize=(26, 5.5 * n + 1.0))
    gs  = gridspec.GridSpec(
        n + 1, 6,
        width_ratios  = col_widths,
        height_ratios = [1] * n + [0.045],
        hspace=0.06, wspace=0.06,
        left=0.07, right=0.98, top=0.94, bottom=0.02,
    )

    axes     = np.array([[fig.add_subplot(gs[r, c]) for c in [0,1,2,4,5]] for r in range(n)])
    cbar_axs = [fig.add_subplot(gs[n, c]) for c in [0,1,2,4,5]]
    spacers  = [fig.add_subplot(gs[r, 3]) for r in range(n + 1)]
    for sp in spacers:
        sp.set_visible(False)

    fig.suptitle(
        f"Model comparison: {label_a} vs. {label_b}  —  "
        f"Translation equivariance test (expected shift: {EXPECTED_SHIFT:.1f} px)",
        fontsize=13, fontweight="bold",
    )

    # Column headers
    headers = [
        "NAC Image",
        f"{label_a}\nSaliency (centered)",
        f"{label_a}\nSaliency (shifted)",
        f"{label_b}\nSaliency (centered)",
        f"{label_b}\nSaliency (shifted)",
    ]
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=10, pad=8, fontweight="bold", linespacing=1.4)

    # Shared saliency scale across all maps
    all_sal = []
    for s in samples:
        for sc, ss in s.per_model.values():
            all_sal += [sc, ss]
    sal_vmin = min(a.min() for a in all_sal)
    sal_vmax = max(a.max() for a in all_sal)

    im_sal = None
    for row, sample in enumerate(samples):
        sc_a, ss_a = sample.per_model[label_a]
        sc_b, ss_b = sample.per_model[label_b]

        # Raw image
        axes[row, 0].imshow(sample.img_norm, cmap="gray")

        # Model A
        im_sal = axes[row, 1].imshow(sc_a, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)
        im_sal = axes[row, 2].imshow(ss_a, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)

        # Model B
        axes[row, 3].imshow(sc_b, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)
        axes[row, 4].imshow(ss_b, cmap="magma", vmin=sal_vmin, vmax=sal_vmax)

        # Peak markers
        for sal, ax in [(sc_a, axes[row,1]), (ss_a, axes[row,2]),
                         (sc_b, axes[row,3]), (ss_b, axes[row,4])]:
            pk = np.unravel_index(np.argmax(sal), sal.shape)
            ax.plot(pk[1], pk[0], "+", color="cyan", ms=13, mew=2)

        # Per-model equivariance annotation on shifted panels
        for ax, model_label, sal_c, sal_s in [
            (axes[row, 2], label_a, sc_a, ss_a),
            (axes[row, 4], label_b, sc_b, ss_b),
        ]:
            err   = sample.per_model_error[model_label]
            rel   = err / EXPECTED_SHIFT * 100
            obs   = compute_peak_shift(sal_c, sal_s)
            color = "#27ae60" if rel < 30 else "#e74c3c"
            ax.text(
                0.03, 0.97,
                f"{obs:.1f} px  (err {rel:.0f}%)",
                transform=ax.transAxes,
                fontsize=7, va="top", ha="left", family="monospace", color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8, alpha=0.92),
                zorder=5,
            )

        # Row label + artifact badge
        row_label = f"† {sample.stem}" if sample.is_artifact else sample.stem
        axes[row, 0].set_ylabel(row_label, fontsize=7, labelpad=5,
                                 color="#c0392b" if sample.is_artifact else "black")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Thin red border on entire row for artifact samples
        if sample.is_artifact:
            for ax in axes[row]:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#c0392b")
                    spine.set_linewidth(1.8)

    # Vertical separator line between model halves
    fig.add_artist(plt.Line2D(
        [col_widths[0]/sum(col_widths) + col_widths[1]/sum(col_widths) +
         col_widths[2]/sum(col_widths) + col_widths[3]/sum(col_widths)/2 + 0.07,
         col_widths[0]/sum(col_widths) + col_widths[1]/sum(col_widths) +
         col_widths[2]/sum(col_widths) + col_widths[3]/sum(col_widths)/2 + 0.07],
        [0.03, 0.93],
        transform=fig.transFigure,
        color="#bbbbbb", linewidth=1.2, linestyle="--",
    ))

    # Colorbars: hide raw col, one bar shared across saliency cols
    cbar_axs[0].set_visible(False)
    for i in [2, 3]:
        cbar_axs[i].set_visible(False)

    for cax in [cbar_axs[1], cbar_axs[4]]:
        cb = fig.colorbar(im_sal, cax=cax, orientation="horizontal")
        cb.set_label("Saliency (shared scale 0–1)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # Legend
    legend_handles = [
        Line2D([0],[0], marker="+", color="cyan", ms=10, mew=2,
               linestyle="none", label="Peak activation"),
        plt.Rectangle((0,0), 1, 1, fc="none", ec="#c0392b", lw=1.8,
                       label="† NAC readout artifact"),
    ]
    fig.legend(handles=legend_handles, loc="lower left", fontsize=9,
               bbox_to_anchor=(0.01, 0.0), framealpha=0.88)

    plt.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_comparison() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("Loading models ...")
    models = {cfg.label: load_model(cfg, device) for cfg in MODELS}

    all_pits = list(Path("data/processed/dataset/test/pits").glob("*.npy"))
    if not all_pits:
        print("No test pits found.")
        return

    print(f"Computing saliency for {len(all_pits)} samples across {len(MODELS)} models ...")
    all_pairs = collect_pairs(all_pits, models, device)
    samples   = select_samples(all_pairs, N_SAMPLES)

    print(f"Selected samples: {[s.stem for s in samples]}")
    print(f"Artifact sample:  {next((s.stem for s in samples if s.is_artifact), 'none')}")

    output_path = build_output_name(MODELS, "luna_fig_model_comparison")
    plot_comparison(samples, [cfg.label for cfg in MODELS], output_path)


if __name__ == "__main__":
    run_comparison()