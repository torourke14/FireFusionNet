from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

def plot_training_loss(train_losses: list[float], save_path: str | None = None):
    """
    Plot training loss vs epoch.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss vs epoch")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def to_numpy_grid(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a torch tensor or numpy array to a 2D numpy array [H, W].
    Accepts shapes:
      - [H, W]
      - [1, H, W]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.array(x)

    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"Expected grid of shape [H, W] or [1, H, W], got {x.shape}")


def reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    title: str = "Reliability diagram",
    save_path: str | None = None
):
    """
    Compute and plot a reliability diagram (calibration curve).
    probs:  [N] tensor of predicted probabilities in [0,1]
    labels: [N] tensor of {0,1}
    """
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(probs, bin_edges, right=True) - 1
    # bin_indices in [0, num_bins-1]

    bin_centers = []
    avg_pred_in_bin = []
    frac_pos_in_bin = []
    counts_in_bin = []

    for b in range(num_bins):
        mask = (bin_indices == b)
        if not np.any(mask):
            continue

        probs_b = probs[mask]
        labels_b = labels[mask]

        bin_center = 0.5 * (bin_edges[b] + bin_edges[b+1])
        bin_centers.append(bin_center)
        avg_pred_in_bin.append(probs_b.mean())
        frac_pos_in_bin.append(labels_b.mean())
        counts_in_bin.append(mask.sum())

    bin_centers = np.array(bin_centers)
    avg_pred_in_bin = np.array(avg_pred_in_bin)
    frac_pos_in_bin = np.array(frac_pos_in_bin)

    plt.figure()
    # y = x diagonal
    plt.plot([0, 1], [0, 1], linestyle="--")

    # calibration curve
    plt.plot(avg_pred_in_bin, frac_pos_in_bin, marker="o")

    # Optional: show bin points scaled by sample count
    # sizes = np.array(counts_in_bin) / np.max(counts_in_bin) * 200
    # plt.scatter(avg_pred_in_bin, frac_pos_in_bin, s=sizes)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical fire frequency")
    plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_XY_grid(
    grid_2d: np.ndarray,
    water_mask: torch.Tensor | np.ndarray | None = None,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None
):
    """
    Plot a heatmap of a continuous field [H, W] with water cells in black.
    Uses 'coolwarm' (blue to red) as base colormap.
    """
    data = np.array(grid_2d, dtype=float)

    if water_mask is not None:
        if isinstance(water_mask, torch.Tensor):
            water_mask = water_mask.detach().cpu().numpy()
        water_mask = np.array(water_mask).astype(bool)
        # mask out water cells
        data = np.ma.masked_where(water_mask, data)

    cmap = plt.cm.get_cmap("coolwarm").copy()
    # any masked entries (water) will be shown as black
    cmap.set_bad(color="black")

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    plt.figure()
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Value")
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_feature_time_XY(
    features: torch.Tensor | np.ndarray,
    water_mask: torch.Tensor | np.ndarray | None,
    time_idx: int | None,
    channel_idx: int,
    title: str = "",
    save_path: str | None = None
):
    """
    Visualize a specific feature channel at time T as a heatmap.
    features: [T, C, H, W], [C, H, W], or [H, W].
    water_mask: [H, W] or None.
    """
    grid_2d = select_time_channel(features, time_idx=time_idx, channel_idx=channel_idx)

    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    grid = np.array(grid)

    if grid.ndim == 2:
        # already [H, W]
        return grid

    if grid.ndim == 3:
        # assume [C, H, W]
        if channel_idx is None:
            raise ValueError("channel_idx must be provided for grid with shape [C, H, W]")
        return grid[channel_idx]

    if grid.ndim == 4:
        # assume [T, C, H, W]
        if time_idx is None or channel_idx is None:
            raise ValueError("time_idx and channel_idx must be provided for [T, C, H, W]")
        return grid[time_idx, channel_idx]

    raise ValueError(f"Unsupported grid shape: {grid.shape}")



def plot_model_prob_grid_time_t(
    probs: torch.Tensor,
    water_mask: torch.Tensor | np.ndarray | None = None,
    title: str = "Predicted ignition probability",
    save_path: str | None = None
):
    """
    Plot model output probabilities [H, W] with water in black.
    probs in [0, 1].
    """
    x = probs.detach().cpu().numpy()
    grid_2d = to_numpy_grid(probs)

    plot_grid(
        grid_2d,
        water_mask=water_mask,
        title=title,
        vmin=0.0,
        vmax=1.0,
        save_path=save_path
    )


def plot_label_grid_time_t(
    labels: torch.Tensor | np.ndarray,
    water_mask: torch.Tensor | np.ndarray | None = None,
    time_idx: int | None = None,
    title: str = "Fire labels",
    save_path: str | None = None):
    """
    Plot binary labels as a discrete heatmap:
      - 0 (no fire) = blue
      - 1 (fire)    = red
      - water       = black
    labels can be:
      - [T, H, W]
      - [H, W]
      - [1, H, W]
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.array(labels)

    if labels.ndim == 3:
        # assume [T, H, W]
        if time_idx is None:
            raise ValueError("time_idx must be provided for label grid with shape [T, H, W]")
        grid_2d = labels[time_idx]
    elif labels.ndim == 2:
        grid_2d = labels
    elif labels.ndim == 3 and labels.shape[0] == 1:
        grid_2d = labels[0]
    else:
        raise ValueError(f"Unsupported labels shape: {labels.shape}")

    grid_2d = grid_2d.astype(float)

    if water_mask is not None:
        if isinstance(water_mask, torch.Tensor):
            water_mask = water_mask.detach().cpu().numpy()
        water_mask = np.array(water_mask).astype(bool)
        grid_2d = np.ma.masked_where(water_mask, grid_2d)

    # discrete colormap: 0 -> blue, 1 -> red, masked -> black
    cmap = ListedColormap(["blue", "red"])
    cmap.set_bad(color="black")

    plt.figure()
    im = plt.imshow(grid_2d, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[0, 1])
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_probability_surface_3d(
    probs_2d: torch.Tensor | np.ndarray,
    title: str = "Ignition probability surface",
    save_path: str | None = None
):
    """ probs_2d: [H, W] probability grid in [0,1] """
    if isinstance(probs_2d, torch.Tensor):
        probs_2d = probs_2d.detach().cpu().numpy()
    probs_2d = np.array(probs_2d)

    H, W = probs_2d.shape
    xs = np.arange(W)
    ys = np.arange(H)
    X, Y = np.meshgrid(xs, ys)

    Z = probs_2d  # [H, W]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

    ax.set_xlabel("X (col)")
    ax.set_ylabel("Y (row)")
    ax.set_zlabel("P(fire)")
    ax.set_title(title)
    ax.view_init(elev=30, azim=225)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()