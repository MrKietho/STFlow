import os
import tempfile
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from torch_cluster import knn_graph
import wandb
import torch
from torch import Tensor
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from torchdiffeq import odeint
from torch_geometric.data import Data
import imageio
from similaritymeasures import frechet_dist
import seaborn as sns


def mean_timestep_displacement_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes the average displacement error between each trajectory per time step."""

    return np.mean(np.mean(np.linalg.norm(X - Y, axis=3), axis=2), axis=0)


def final_displacement_error(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes the mean final displacement error of the last timestep over all trajectories."""

    final_displacement_errors = np.linalg.norm(X[:, -1, :, :] - Y[:, -1, :, :], axis=2)
    mean_trajectory_fdes = np.nanmean(final_displacement_errors, axis=1)
    return mean_trajectory_fdes


def relative_final_displacement_error(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes the final displacement errors and weigh them by the ground truth cumulative distances."""
    final_displacement_errors = final_displacement_error(X, Y)

    distances = np.diff(X, axis=1)
    cum_distances = np.sum(np.linalg.norm(distances, axis=3), axis=1)
    mean_cum_distances = np.nanmean(cum_distances, axis=1)

    return final_displacement_errors / mean_cum_distances


def cumulative_displacement_error(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes the mean cumulative displacement error over all trajectories."""

    displacement_errors = np.linalg.norm(X - Y, axis=3)
    cumulative_displacement_errors = np.nansum(displacement_errors, axis=1)
    mean_trajectory_cdes = np.nanmean(cumulative_displacement_errors, axis=1)
    return mean_trajectory_cdes


def one_step_displacement_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes the mean displacement error between each trajectory per time step."""

    displacement_errors = np.linalg.norm(X - Y, axis=3)
    mean_trajectory = np.nanmean(displacement_errors, axis=1)
    mean_nodes = np.nanmean(mean_trajectory, axis=1)

    return mean_nodes


def mean_square_displacement(X: np.ndarray, time_index: list) -> np.ndarray:
    """Computer the mean square displacement for each timestep, averaged over all trajectories"""

    msds = np.zeros(len(time_index))

    for i, t in enumerate(time_index):
        displacement_errors = np.linalg.norm(X[:, 0] - X[:, t], axis=1)
        msds[i] = np.nanmean(displacement_errors)

    return msds


def velocity_autocorrelation(X: np.ndarray, time_lags: list) -> tuple[np.ndarray, np.ndarray]:
    """Computer the velocity autocorrelation for each timestep, averaged over all trajectories"""

    # Calculate velocities (differences in position between consecutive time steps)
    velocities = np.diff(X, axis=1)  # Shape: (num_trajectories, num_timesteps - 1, 2)

    # Initialize an array to store autocorrelation values for each time lag
    vel_corrs = np.zeros(len(time_lags))
    vel_corrs_std = np.zeros(len(time_lags))

    normalization = np.mean(np.einsum("ijk,ijk->ij", velocities, velocities))

    for k, lag in enumerate(time_lags):
        if lag >= velocities.shape[1]:  # Skip if lag is too large
            vel_corrs[k] = np.nan
            continue

        # Dot product of velocity vectors with their lagged counterparts
        # Shape of dot_products: (num_trajectories, num_timesteps - 1 - lag)
        if lag == 0:
            dot_products = np.einsum("ijk,ijk->ij", velocities, velocities)
        else:
            dot_products = np.einsum(
                "ijk,ijk->ij", velocities[:, :-lag, :], velocities[:, lag:, :]
            )

        # Average over all trajectories and time steps for this lag
        vel_corrs[k] = np.mean(dot_products) / normalization
        vel_corrs_std[k] = np.std(dot_products) / normalization

    return vel_corrs, vel_corrs_std


def calculate_tortuosity(X: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of tortuosity for multiple trajectories.

    Parameters:
    trajectories: ndarray of shape (n_trajectories, frames, 2)

    Returns:
    ndarray of shape (n_trajectories,) containing the tortuosity of each trajectory
    """
    # Calculate differences between consecutive points for all trajectories
    differences = np.diff(X, axis=1)  # Shape: (n_trajectories, frames-1, 2)

    # Calculate segment lengths for all trajectories
    segment_lengths = np.sqrt(
        np.sum(differences**2, axis=2)
    )  # Shape: (n_trajectories, frames-1)

    # Sum up segment lengths for total path length
    total_path_lengths = np.sum(segment_lengths, axis=1)  # Shape: (n_trajectories,)

    # Calculate straight-line distances between start and end points
    start_points = X[:, 0, :]  # Shape: (n_trajectories, 2)
    end_points = X[:, -1, :]  # Shape: (n_trajectories, 2)
    straight_line_distances = np.sqrt(np.sum((end_points - start_points) ** 2, axis=1))

    # Handle potential division by zero and NaN values and minimize distance travelled
    mask = (
        (straight_line_distances != 0)
        & ~np.isnan(straight_line_distances)
        & (straight_line_distances >= 5)
    )
    tortuosities = np.ones(len(X))  # Initialize with 1.0
    tortuosities[mask] = total_path_lengths[mask] / straight_line_distances[mask]

    # Filter out any remaining NaN values
    valid_tortuosities = tortuosities[~np.isnan(tortuosities)]

    return valid_tortuosities


def calculate_neighbourhood_velocity_correlation(
    positions: Tensor, knn: int = 10
) -> np.ndarray:
    """Calculate correlation between each trajectory velocity and its kn neighbours

    Args:
        positions: ndarray of shape (n_trajectories, frames, 2)
        knn: number of nearest neighbours to consider
    """

    corrs = []
    edges_over_time = []
    velocities = torch.diff(positions, dim=1)
    velocities = velocities / (torch.norm(velocities, dim=-1, keepdim=True) + 1e-8)

    for frame in range(velocities.shape[1]):
        edges = knn_graph(positions[:, frame], k=knn)
        edges_over_time.append(edges)

    for node in range(velocities.shape[0]):
        node_vels = velocities[node].numpy()
        neighbour_vels = torch.nan * torch.zeros((knn, velocities.shape[1], 2))

        for frame in range(velocities.shape[1]):
            neighbours = edges_over_time[frame][0][edges_over_time[frame][1] == node]
            for i, neighbour in enumerate(neighbours):
                neighbour_vels[i, frame] = velocities[neighbour, frame]

        neighbour_vels = neighbour_vels.numpy()
        for neighbour in range(knn):  # If neighbour contains nan, skip
            if np.isnan(neighbour_vels[neighbour]).any():
                continue
            corr_1 = np.corrcoef(node_vels[:, 0], neighbour_vels[neighbour, :, 0])[0, 1]
            corr_2 = np.corrcoef(node_vels[:, 1], neighbour_vels[neighbour, :, 1])[0, 1]
            corr_3 = np.corrcoef(
                np.linalg.norm(node_vels, axis=1),
                np.linalg.norm(neighbour_vels[neighbour], axis=1),
            )[0, 1]
            corr = [np.nan_to_num(corr_1), np.nan_to_num(corr_2), np.nan_to_num(corr_3)]
            corrs.append(corr)

    return np.array(corrs)


def calculate_neighbour_dist_over_time(
    trajectories: Tensor, knn: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate change in mean distances between neighbouring nodes over time

    Args:
        trajectories: ndarray of shape (n_trajectories, frames, 2)
    """

    distances, stds = [], []
    for frame in range(trajectories.shape[1]):
        pos = trajectories[:, frame]

        if len(pos) == 1:
            continue
        dist = pdist(pos, metric="euclidean")
        # Get knn distances
        knn_dists = np.sort(dist.flatten())[:knn]  # type: ignore
        distances.append(knn_dists.mean())
        stds.append(knn_dists.std())

    return np.array(distances), np.array(stds)


def calculate_velocity_magnitude(prior: Tensor, predictions: Tensor, ground_truth: Tensor):
    """Calculate velocitiy magnitude for prior, predictions and ground truth trajectories"""
    prior_vels = np.diff(prior.cpu().numpy(), axis=1)
    gt_vels = np.diff(ground_truth.cpu().numpy(), axis=1)
    pred_vels = np.diff(predictions.cpu().numpy(), axis=1)

    # Handle potential NaN values
    prior_vels[np.isnan(prior_vels)] = 0
    gt_vels[np.isnan(gt_vels)] = 0
    pred_vels[np.isnan(pred_vels)] = 0

    # Calculate velocities
    prior_vels = np.linalg.norm(prior_vels, axis=2)
    gt_vels = np.linalg.norm(gt_vels, axis=2)
    pred_vels = np.linalg.norm(pred_vels, axis=2)

    return prior_vels, pred_vels, gt_vels


def calculate_acceleration(prior: Tensor, predictions: Tensor, ground_truth: Tensor):
    """Calculate accelerations for prior, predictions and ground truth trajectories"""
    prior_accs = np.diff(np.diff(prior.cpu().numpy(), axis=1), axis=1)
    gt_accs = np.diff(np.diff(ground_truth.cpu().numpy(), axis=1), axis=1)
    pred_accs = np.diff(np.diff(predictions.cpu().numpy(), axis=1), axis=1)

    # Handle potential NaN values
    prior_accs[np.isnan(prior_accs)] = 0
    gt_accs[np.isnan(gt_accs)] = 0
    pred_accs[np.isnan(pred_accs)] = 0

    # Calculate accelerations
    prior_accs = np.linalg.norm(prior_accs, axis=2)
    gt_accs = np.linalg.norm(gt_accs, axis=2)
    pred_accs = np.linalg.norm(pred_accs, axis=2)

    return prior_accs, pred_accs, gt_accs


def calculate_angular_velocity(prior: Tensor, predictions: Tensor, ground_truth: Tensor):
    """Calculate angular velocities for prior, predictions and ground truth trajectories"""
    prior_ang_vels = np.diff(prior.cpu().numpy(), axis=1)
    gt_ang_vels = np.diff(ground_truth.cpu().numpy(), axis=1)
    pred_ang_vels = np.diff(predictions.cpu().numpy(), axis=1)

    # Calculate angles
    prior_θ = np.arctan2(prior_ang_vels[:, :, 1], prior_ang_vels[:, :, 0])
    gt_θ = np.arctan2(gt_ang_vels[:, :, 1], gt_ang_vels[:, :, 0])
    pred_θ = np.arctan2(pred_ang_vels[:, :, 1], pred_ang_vels[:, :, 0])

    # Unwrap angles
    prior_θ = np.unwrap(prior_θ)
    gt_θ = np.unwrap(gt_θ)
    pred_θ = np.unwrap(pred_θ)

    # Calculate angular velocities
    prior_ω = np.diff(prior_θ, axis=1)
    gt_ω = np.diff(gt_θ, axis=1)
    pred_ω = np.diff(pred_θ, axis=1)

    return prior_ω, pred_ω, gt_ω


def calculate_nearest_neighbour_dist(x: np.ndarray, config: dict) -> np.ndarray:
    """Calculate the nearest neighbour distance for each point in x
    x: np.array of shape (n_trajs, frames, 2)
    """

    distances = []
    if x.shape[0] == 1:
        return np.array(distances)

    filter_dist = (
        config["radius_graph"] if config["radius_graph"] is not None else 1
    )  # m (or pixels)

    for frame in range(x.shape[1]):
        pairs = squareform(pdist(x[:, frame], metric="euclidean"))
        np.fill_diagonal(pairs, np.inf)
        min_dists = np.min(pairs, axis=1)
        filtered_dists = min_dists[min_dists < filter_dist]
        distances.extend(list(filtered_dists))

    return np.array(distances)


def plot_local_velocity_correlation(
    predictions: np.ndarray, ground_truth: np.ndarray, prior: np.ndarray
) -> plt.Figure:
    "Plot the local velocity x, y and magnitude correlations between predictions and ground truth"

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for ax, idx in zip(axs, range(3)):
        ax.violinplot(
            [prior[:, idx], predictions[:, idx], ground_truth[:, idx]],
            positions=[0, 1, 2],
            showmeans=False,
            showmedians=False,
        )
        bp = ax.boxplot(
            [prior[:, idx], predictions[:, idx], ground_truth[:, idx]],
            positions=[0, 1, 2],
            widths=0.10,
            showfliers=False,
            patch_artist=True,
        )
        # Customize box plots
        colors = ["tab:orange", "tab:blue", "tab:red"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks([0, 1, 2], ["Prior", "Predictions", "Ground Truth"])
        ax.grid(True, linestyle=":", alpha=0.5)

    axs[0].set_ylabel("Correlation")
    axs[0].set_title("X Velocity")
    axs[1].set_title("Y Velocity")
    axs[2].set_title("Magnitude")

    fig.suptitle("Local Velocity Correlation")

    plt.tight_layout()

    return fig


def make_violin_box_plot(
    prior: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    min: float = -np.inf,
    max: float = np.inf,
) -> plt.Figure:
    """
    Create a violin plot comparing the distributions of BOT values for predictions and ground truth.
    Parameters:
    ground_truth: ndarray of shape (n_trajectories, frames, 2)
    predictions: ndarray of shape (n_trajectories, frames, 2)
    title: str, title of the plot

    Returns:
    matplotlib figure object"""

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot violin plots
    parts = ax.violinplot(
        [prior, pred, gt],
        positions=[0, 1, 2],
        showmeans=False,
        showmedians=False,
    )

    # Customize violin plots
    for pc in parts["bodies"]:  # type: ignore
        pc.set_alpha(0.3)

    # Plot box plots inside violin plots
    bp = ax.boxplot(
        [prior, pred, gt],
        positions=[0, 1, 2],
        widths=0.10,
        showfliers=False,
        patch_artist=True,
    )

    # Customize box plots
    colors = ["tab:orange", "tab:blue", "tab:red"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Add individual points with jitter
    for idx, (data, pos) in enumerate(zip([prior, pred, gt], [0, 1, 2])):
        noise = np.random.normal(0, 0.01, size=len(data))
        ax.scatter(noise + pos, data, alpha=0.2, s=5, c=[colors[idx]])

    # Customize plot
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Prior", "Predictions", "Ground Truth"])
    ax.set_ylabel(title)
    ax.set_title(f"{title} Distribution Comparison")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Add statistical information
    info_text = (
        f"Prior: μ={np.mean(prior):.4f} ± {np.std(prior):.3f}\n"
        f"Predictions: μ={np.mean(pred):.4f} ± {np.std(pred):.3f}\n"
        f"Ground Truth: μ={np.mean(gt):.4f} ± {np.std(gt):.3f}\n"
    )
    ax.text(
        0.5,
        0.93,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def make_kde_plot(
    prior: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    min: float = -np.inf,
    max: float = np.inf,
) -> plt.Figure:
    """
    Create a KDE plot comparing the distributions.
    Parameters:
    prior: ndarray of shape (n_trajectories, frames, 2)
    predictions: ndarray of shape (n_trajectories, frames, 2)
    ground_truth: ndarray of shape (n_trajectories, frames, 2)
    title: str, title of the plot
    min: float, minimum value for the x-axis
    max: float, maximum value for the x-axis
    """

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot KDE plots
    sns.kdeplot(
        prior.flatten(),
        color="tab:orange",
        label="Prior",
        ax=ax,
        fill=True,
        common_norm=False,
        clip=(min, max),
    )
    sns.kdeplot(
        pred.flatten(),
        color="tab:blue",
        label="Predictions",
        ax=ax,
        fill=True,
        common_norm=False,
        clip=(min, max),
    )
    sns.kdeplot(
        gt.flatten(),
        color="tab:red",
        label="Ground Truth",
        ax=ax,
        fill=True,
        common_norm=False,
        clip=(min, max),
    )

    # Customize plot
    ax.set_xlabel(title)
    ax.set_ylabel("Density")
    ax.set_title(f"{title} Distribution Comparison")
    # ax.grid(True, linestyle=":", alpha=0.5)

    ax.legend()

    # Add statistical information
    info_text = (
        f"Prior: μ={np.mean(prior):.4f} ± {np.std(prior):.3f}\n"
        f"Predictions: μ={np.mean(pred):.4f} ± {np.std(pred):.3f}\n"
        f"Ground Truth: μ={np.mean(gt):.4f} ± {np.std(gt):.3f}\n"
    )
    ax.text(
        0.83,
        0.88,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def make_hist_plot(
    pred: np.ndarray, gt: np.ndarray, title: str, prior: np.ndarray | None = None, min_max=None
):
    """
    Plots two overlaid histograms comparing distributions of pred and gt.

    Parameters:
    - pred: np.ndarray : Predicted values
    - gt: np.ndarray : Ground truth values
    - title: str : Title of the plot
    """
    fig = plt.figure(figsize=(8, 6))

    # Filter values between min and max
    if min_max is not None:
        pred = pred[np.logical_and(pred >= min_max[0], pred <= min_max[1])]
        gt = gt[np.logical_and(gt >= min_max[0], gt <= min_max[1])]
        if prior is not None:
            prior = prior[np.logical_and(prior >= min_max[0], prior <= min_max[1])]

    # Compute statistics
    mean_pred, var_pred = np.mean(pred), np.var(pred)
    mean_gt, var_gt = np.mean(gt), np.var(gt)

    global_min = min(pred.min(), gt.min())
    global_max = max(pred.max(), gt.max())

    bins = np.linspace(global_min, global_max, 50)

    # Compute bin indices
    pred_counts, bin_edges = np.histogram(pred.flatten(), bins)
    gt_counts, _ = np.histogram(gt.flatten(), bins)
    if prior is not None and len(prior) > 0:
        prior_counts, _ = np.histogram(prior.flatten(), bins)

    bin_widths = np.diff(bin_edges)

    # Scatter plot to represent densities with dots
    x_bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    y_pred = (
        pred_counts / (pred.size * bin_widths) if pred.size > 0 else np.zeros_like(pred_counts)
    )
    y_gt = gt_counts / (gt.size * bin_widths) if gt.size > 0 else np.zeros_like(gt_counts)
    if prior is not None and len(prior) > 0:
        y_prior = prior_counts / (prior.size * bin_widths)

    # Draw small lines between the two histograms points
    plt.vlines(
        x=x_bins,
        ymin=y_pred,
        ymax=y_gt,
        color="black",
        linestyle="dotted",
        alpha=0.6,
        linewidth=0.5,
    )

    # Plot histograms
    plt.scatter(x_bins, y_pred, color="tab:blue", alpha=0.9, s=50, label="Prediction")
    plt.scatter(x_bins, y_gt, color="tab:red", alpha=0.9, s=50, label="Ground Truth")
    if prior is not None:
        plt.scatter(x_bins, y_prior, color="tab:orange", alpha=0.8, s=18, label="Prior")

    # Add a small text box with statistics
    textstr = (
        f"Pred: μ={mean_pred:.2f}, σ²={var_pred:.2f}\nGT: μ={mean_gt:.2f}, σ²={var_gt:.2f}"
    )
    plt.text(
        0.54,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, color="white"),
        alpha=0.8,
    )

    # Labels and legend
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.title(f"{title} Density Comparison (N={gt.shape[0]})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    return fig


def make_log_hist_plot(
    pred: np.ndarray, gt: np.ndarray, title: str, prior: np.ndarray | None = None, min_max=None
) -> plt.Figure:
    """
    Creates a log-scaled histogram plot of the prediction and ground truth data.

    Parameters
    ----------
    pred : np.ndarray
        The predicted data.
    gt : np.ndarray
        The ground truth data.
    title : str
    """
    fig = plt.figure(figsize=(8, 6))

    # Filter values between min and max
    if min_max is not None:
        pred = pred[np.logical_and(pred >= min_max[0], pred <= min_max[1])]
        gt = gt[np.logical_and(gt >= min_max[0], gt <= min_max[1])]
        if prior is not None:
            prior = prior[np.logical_and(prior >= min_max[0], prior <= min_max[1])]

    # Compute statistics
    mean_pred, var_pred = np.mean(pred), np.var(pred)
    mean_gt, var_gt = np.mean(gt), np.var(gt)

    global_min = min(pred.min(), gt.min())
    global_max = max(pred.max(), gt.max())

    bins = np.linspace(global_min, global_max, 50)

    # Compute bin indices
    pred_counts, bin_edges = np.histogram(pred.flatten(), bins)
    gt_counts, _ = np.histogram(gt.flatten(), bins)
    if prior is not None:
        prior_counts, _ = np.histogram(prior.flatten(), bins)

    bin_widths = np.diff(bin_edges)

    # Scatter plot to represent densities with dots
    x_bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    y_pred = (
        pred_counts / (pred.size * bin_widths) if pred.size > 0 else np.zeros_like(pred_counts)
    )
    y_gt = gt_counts / (gt.size * bin_widths) if gt.size > 0 else np.zeros_like(gt_counts)
    if prior is not None and len(prior) > 0:
        y_prior = prior_counts / (prior.size * bin_widths)

    # Draw small lines between the two histograms points
    plt.vlines(
        x=x_bins,
        ymin=y_pred,
        ymax=y_gt,
        color="black",
        linestyle="--",
        alpha=0.6,
        linewidth=0.5,
    )

    # Plot histograms
    plt.scatter(x_bins, y_pred, color="tab:blue", alpha=0.9, s=60, label="Prediction")
    plt.scatter(x_bins, y_gt, color="tab:red", alpha=0.9, s=60, label="Ground Truth")
    if prior is not None:
        plt.scatter(x_bins, y_prior, color="tab:orange", alpha=0.8, s=18, label="Prior")

    # Set to log scale
    plt.yscale("log")

    # Add a small text box with statistics
    textstr = (
        f"Pred: μ={mean_pred:.2f}, σ²={var_pred:.2f}\nGT: μ={mean_gt:.2f}, σ²={var_gt:.2f}"
    )
    plt.text(
        0.55,
        0.92,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, color="white"),
        alpha=0.8,
    )

    # Labels and legend
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.title(f"{title} Log Density Comparison (N={gt.shape[0]})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    return fig


def make_hist_plot_paper(
    pred: np.ndarray, gt: np.ndarray, title: str, prior: np.ndarray | None = None, min_max=None
):
    """
    Plots two overlaid histograms comparing distributions of pred and gt.

    Parameters:
    - pred: np.ndarray : Predicted values
    - gt: np.ndarray : Ground truth values
    - title: str : Title of the plot
    """
    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )
    fig = plt.figure(figsize=(5, 4))

    # Filter values between min and max
    if min_max is not None:
        pred = pred[np.logical_and(pred >= min_max[0], pred <= min_max[1])]
        gt = gt[np.logical_and(gt >= min_max[0], gt <= min_max[1])]
        if prior is not None:
            prior = prior[np.logical_and(prior >= min_max[0], prior <= min_max[1])]

    # Compute statistics
    mean_pred, var_pred = np.mean(pred), np.var(pred)
    mean_gt, var_gt = np.mean(gt), np.var(gt)

    global_min = min(pred.min(), gt.min())
    global_max = max(pred.max(), gt.max())

    bins = np.linspace(global_min, global_max, 30)

    # Compute bin indices
    pred_counts, bin_edges = np.histogram(pred.flatten(), bins)
    gt_counts, _ = np.histogram(gt.flatten(), bins)
    if prior is not None and len(prior) > 0:
        prior_counts, _ = np.histogram(prior.flatten(), bins)

    bin_widths = np.diff(bin_edges)

    # Scatter plot to represent densities with dots
    x_bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    y_pred = (
        pred_counts / (pred.size * bin_widths) if pred.size > 0 else np.zeros_like(pred_counts)
    )
    y_gt = gt_counts / (gt.size * bin_widths) if gt.size > 0 else np.zeros_like(gt_counts)
    if prior is not None and len(prior) > 0:
        y_prior = prior_counts / (prior.size * bin_widths)

    # Draw small lines between the two histograms points
    plt.vlines(
        x=x_bins,
        ymin=y_pred,
        ymax=y_gt,
        color="black",
        linestyle="dotted",
        alpha=0.8,
        linewidth=0.8,
        zorder=-1,
    )

    # Plot histograms
    plt.scatter(
        x_bins,
        y_pred,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=3,
        zorder=2,
        s=80,
        label="Prediction",
    )
    plt.scatter(x_bins, y_gt, color="tab:red", s=70, zorder=1, label="Ground Truth")
    if prior is not None:
        plt.scatter(x_bins, y_prior, color="tab:orange", alpha=0.8, s=18, label="Prior")

    # Add a small text box with statistics
    textstr = (
        f"Pred: μ={mean_pred:.2f}, σ²={var_pred:.2f}\nGT: μ={mean_gt:.2f}, σ²={var_gt:.2f}"
    )
    plt.text(
        0.62,
        0.96,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, color="white"),
        alpha=0.9,
    )
    plt.gca().set_ybound(plt.gca().get_ybound()[0], plt.gca().get_ybound()[1] * 1.15)

    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().tick_params(axis="both", which="major", labelsize=13)

    # Labels and legend
    plt.xlabel(title, fontsize=14)
    plt.ylabel("Density", fontsize=13)
    # plt.title(f"{title} Density Comparison (N={gt.shape[0]})")
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig


def make_hist_log_plot_paper(
    pred: np.ndarray, gt: np.ndarray, title: str, prior: np.ndarray | None = None, min_max=None
):
    """
    Plots two overlaid histograms comparing distributions of pred and gt.

    Parameters:
    - pred: np.ndarray : Predicted values
    - gt: np.ndarray : Ground truth values
    - title: str : Title of the plot
    """
    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )
    fig = plt.figure(figsize=(5, 4))

    plt.yscale("log")

    # Filter values between min and max
    if min_max is not None:
        pred = pred[np.logical_and(pred >= min_max[0], pred <= min_max[1])]
        gt = gt[np.logical_and(gt >= min_max[0], gt <= min_max[1])]
        if prior is not None:
            prior = prior[np.logical_and(prior >= min_max[0], prior <= min_max[1])]

    # Compute statistics
    mean_pred, var_pred = np.mean(pred), np.var(pred)
    mean_gt, var_gt = np.mean(gt), np.var(gt)

    global_min = min(pred.min(), gt.min())
    global_max = max(pred.max(), gt.max())

    bins = np.linspace(global_min, global_max, 30)

    # Compute bin indices
    pred_counts, bin_edges = np.histogram(pred.flatten(), bins)
    gt_counts, _ = np.histogram(gt.flatten(), bins)
    if prior is not None and len(prior) > 0:
        prior_counts, _ = np.histogram(prior.flatten(), bins)

    bin_widths = np.diff(bin_edges)

    # Scatter plot to represent densities with dots
    x_bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    y_pred = (
        pred_counts / (pred.size * bin_widths) if pred.size > 0 else np.zeros_like(pred_counts)
    )
    y_gt = gt_counts / (gt.size * bin_widths) if gt.size > 0 else np.zeros_like(gt_counts)
    if prior is not None and len(prior) > 0:
        y_prior = prior_counts / (prior.size * bin_widths)

    # Draw small lines between the two histograms points
    plt.vlines(
        x=x_bins,
        ymin=y_pred,
        ymax=y_gt,
        color="black",
        linestyle="dotted",
        alpha=0.8,
        linewidth=0.8,
        zorder=-1,
    )

    # Plot histograms
    plt.scatter(
        x_bins,
        y_pred,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=3,
        # alpha=0.9,
        s=80,
        label="Prediction",
        zorder=2,
    )
    plt.scatter(x_bins, y_gt, color="tab:red", s=70, label="Ground Truth", zorder=1)
    if prior is not None:
        plt.scatter(x_bins, y_prior, color="tab:orange", alpha=0.8, s=18, label="Prior")

    # Add a small text box with statistics
    textstr = (
        f"Pred: μ={mean_pred:.2f}, σ²={var_pred:.2f}\nGT: μ={mean_gt:.2f}, σ²={var_gt:.2f}"
    )
    plt.text(
        0.62,
        0.96,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, color="white"),
        alpha=0.9,
    )
    plt.gca().set_ybound(plt.gca().get_ybound()[0], plt.gca().get_ybound()[1] * 1.15)

    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().tick_params(axis="both", which="major", labelsize=13)

    # Labels and legend
    plt.xlabel(title, fontsize=14)
    plt.ylabel("Log Density", fontsize=13)
    # plt.title(f"{title} Density Comparison (N={gt.shape[0]})")
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig


def make_hist_plot_all_paper(pred, gt, titles):
    """
    Plots two overlaid histograms comparing distributions of pred and gt.

    Parameters:
    - pred: np.ndarray : Predicted values
    - gt: np.ndarray : Ground truth values
    - title: str : Title of the plot
    """
    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, figure=fig)

    for i in range(6):
        ax = fig.add_subplot(gs[i])

        # Compute statistics
        mean_pred, var_pred = np.mean(pred[i]), np.var(pred[i])
        mean_gt, var_gt = np.mean(gt[i]), np.var(gt[i])

        global_min = min(pred[i].min(), gt[i].min())
        global_max = max(pred[i].max(), gt[i].max())

        bins = np.linspace(global_min, global_max, 30)

        # Compute bin indices
        pred_counts, bin_edges = np.histogram(pred[i].flatten(), bins)
        gt_counts, _ = np.histogram(gt[i].flatten(), bins)

        bin_widths = np.diff(bin_edges)

        # Scatter plot to represent densities with dots
        x_bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        y_pred = (
            pred_counts / (pred[i].size * bin_widths)
            if pred[i].size > 0
            else np.zeros_like(pred_counts)
        )
        y_gt = (
            gt_counts / (gt[i].size * bin_widths)
            if gt[i].size > 0
            else np.zeros_like(gt_counts)
        )

        # Draw small lines between the two histograms points
        # ax.vlines(
        #     x=x_bins,
        #     ymin=y_pred,
        #     ymax=y_gt,
        #     color="black",
        #     linestyle="dotted",
        #     alpha=0.8,
        #     linewidth=0.8,
        #     zorder=-1,
        # )

        # Plot histograms
        ax.scatter(
            x_bins,
            y_pred,
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=2,
            # linestyle=":",
            zorder=2,
            s=75,
            label="Prediction",
        )
        # for x, y in zip(x_bins, y_pred):
        #     circle = patches.Ellipse(
        #         (x, y),
        #         width=0.1,
        #         height=0.1,
        #         facecolor="none",
        #         edgecolor="tab:blue",
        #         linewidth=1.5,
        #         zorder=2,
        #         linestyle=":",
        #         transform=ax.transData,
        #     )
        #     ax.add_patch(circle)
        ax.scatter(x_bins, y_gt, color="tab:red", s=75, zorder=1, label="Ground Truth")

        # Add a small text box with statistics
        # textstr = (
        #     f"Pred: μ={mean_pred:.2f}, σ²={var_pred:.2f}\nGT: μ={mean_gt:.2f}, σ²={var_gt:.2f}"
        # )
        # ax.text(
        #     0.5,
        #     0.96,
        #     textstr,
        #     transform=plt.gca().transAxes,
        #     fontsize=11,
        #     verticalalignment="top",
        #     bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, color="white"),
        #     alpha=0.9,
        # )
        # ax.set_ybound(ax.get_ybound()[0], ax.get_ybound()[1] * 1.15)

        if i == 2 or i == 5:
            ax.set_yscale("log")

        ax.spines["top"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        ax.tick_params(axis="both", which="major", labelsize=13)

        # Labels and legend
        ax.set_xlabel(titles[i], fontsize=16)
        if i == 0 or i == 3:
            ax.set_ylabel("Density", fontsize=15)
        if i == 2 or i == 5:
            ax.set_ylabel("Log Density", fontsize=15)
        # plt.title(f"{title} Density Comparison (N={gt.shape[0]})")
        # plt.legend(fontsize=12, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.legend(
        plt.gca().get_legend_handles_labels()[0],
        plt.gca().get_legend_handles_labels()[1],
        loc="upper center",
        bbox_to_anchor=(0.51, 1.01),
        ncol=2,
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    return fig


def plot_velocity_autocorrelation(
    prior_autocorr: np.ndarray,
    x_autocorr: np.ndarray,
    gt_autocorr: np.ndarray,
    time_step: int,
) -> plt.Figure:
    """
    Creates an enhanced plot of velocity autocorrelation over time.

    Parameters:
    -----------
    prior_autocorr: np.ndarray
        Prior velocity autocorrelation values
    x_autocorr: np.ndarray
        Predicted velocity autocorrelation values
    gt_autocorr: np.ndarray
        ground truth velocity autocorrelation values

    Returns:
    --------
    plt.Figure
    """
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])

    # Generate time points
    times = np.array(range(0, (x_autocorr.shape[0]) * time_step, time_step))

    # Main plot
    # Plot confidence intervals
    ax_main.fill_between(
        times,
        prior_autocorr - np.std(prior_autocorr),
        prior_autocorr + np.std(prior_autocorr),
        alpha=0.2,
        color="tab:orange",
        label="Prior 68% CI",
    )

    ax_main.fill_between(
        times,
        x_autocorr - np.std(x_autocorr),
        x_autocorr + np.std(x_autocorr),
        alpha=0.2,
        color="tab:blue",
        label="Pred. 68% CI",
    )
    ax_main.fill_between(
        times,
        gt_autocorr - np.std(gt_autocorr),
        gt_autocorr + np.std(gt_autocorr),
        alpha=0.2,
        color="tab:red",
        label="GT 68% CI",
    )

    # Plot mean lines
    ax_main.plot(
        times, prior_autocorr, color="tab:orange", linestyle="-", linewidth=2, label="Prior"
    )
    ax_main.plot(
        times, x_autocorr, color="tab:blue", linestyle="-", linewidth=2, label="Predictions"
    )
    ax_main.plot(
        times, gt_autocorr, color="tab:red", linestyle="-", linewidth=2, label="Ground Truth"
    )

    # Customize main plot
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Velocity Correlation")
    ax_main.legend(loc="upper right")
    ax_main.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_main.set_title("Velocity Autocorrelation Analysis", pad=20)

    # Calculate and plot difference
    difference = x_autocorr - gt_autocorr
    diff_std = np.sqrt(np.var(x_autocorr) + np.var(gt_autocorr))

    ax_diff.fill_between(
        times, -diff_std, diff_std, alpha=0.2, color="gray", label="Difference 68% CI"
    )
    ax_diff.plot(times, difference, color="black", label="Difference")
    ax_diff.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Customize difference plot
    ax_diff.set_xlabel("Time (Minutes)")
    ax_diff.set_ylabel("Difference")
    ax_diff.grid(True, linestyle=":", alpha=0.5)

    # # Add statistical information
    # corr, p_value = stats.pearsonr(x_autocorr, gt_autocorr)
    # rmse = np.sqrt(np.mean((x_autocorr - gt_autocorr) ** 2))

    # stats_text = f"Correlation: {corr:.3f}\nRMSE: {rmse:.3f}"

    # plt.figtext(
    #     0.02,
    #     0.98,
    #     stats_text,
    #     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    #     verticalalignment="top",
    #     fontsize=9,
    # )

    # plt.tight_layout()

    return fig


def plot_mean_square_displacement(
    prior_msds: np.ndarray,
    msds: np.ndarray,
    gt_msds: np.ndarray,
    time_step: int,
) -> plt.Figure:
    """
    Enhanced plot of mean square displacement over time with motion analysis.

    Parameters:
    -----------
    trajectories: Tensor
        Predicted trajectories
    ground_truth: Tensor
        Ground truth trajectories
    time_step: int
        Time step in minutes
    diffusive: Optional[Tensor]
        Reference diffusive motion trajectories
    ballistic: Optional[Tensor]
        Reference ballistic motion trajectories
    save_path: Optional[str]
        Path to save the figure

    Returns:
    --------
    Tuple[plt.Figure, List]
        Figure and MSD values
    """

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.15)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1])
    ax_alpha = fig.add_subplot(gs[2])
    plt.style.use("seaborn-v0_8-paper")

    # Calculate standard errors
    prior_std = np.std(prior_msds, axis=0)
    pred_std = np.std(msds, axis=0)
    gt_std = np.std(gt_msds, axis=0)

    times = np.array(range(0, msds.shape[0] * time_step, time_step))

    # Main plot with confidence intervals
    ax_main.fill_between(
        times,
        prior_msds - prior_std,
        prior_msds + prior_std,
        alpha=0.2,
        color="tab:orange",
        label="Pred. 68% CI",
    )
    ax_main.fill_between(
        times,
        msds - pred_std,
        msds + pred_std,
        alpha=0.2,
        color="tab:blue",
        label="Pred. 68% CI",
    )
    ax_main.fill_between(
        times,
        gt_msds - gt_std,
        gt_msds + gt_std,
        alpha=0.2,
        color="tab:red",
        label="GT 68% CI",
    )

    # Plot mean lines
    ax_main.plot(times, prior_msds, "tab:orange", linewidth=2, label="Prior")
    ax_main.plot(times, msds, "tab:blue", linewidth=2, label="Predictions")
    ax_main.plot(times, gt_msds, "tab:red", linewidth=2, label="Ground Truth")

    # Add reference lines for different motion types
    time_mask = times > 0  # Avoid log(0)
    ref_times = times[time_mask]

    # Add ballistic reference (α = 2)
    ballistic_ref = ref_times**2
    ax_main.plot(
        ref_times,
        ballistic_ref * msds[time_mask][0] / ballistic_ref[0],
        "k:",
        alpha=0.5,
        label="α = 2 (Ballistic)",
    )

    # Add diffusive reference (α = 1)
    diffusive_ref = ref_times
    ax_main.plot(
        ref_times,
        diffusive_ref * msds[time_mask][0] / diffusive_ref[0],
        "k--",
        alpha=0.5,
        label="α = 1 (Diffusive)",
    )

    # Plot MSD ratio
    ratio = msds / gt_msds.clip(1e-8)
    # ax_ratio.semilogx(times, ratio, "k-", label="MSD Ratio")
    ax_ratio.plot(times, ratio, "k-", label="MSD Ratio")
    ax_ratio.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax_ratio.fill_between(times, 0.5, 1.5, color="gray", alpha=0.2)
    ax_ratio.set_ylabel("Pred/GT Ratio")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.grid(True)

    ax_alpha.axhline(1, color="gray", linestyle="--", alpha=0.5, label="Diffusive")
    ax_alpha.axhline(2, color="gray", linestyle=":", alpha=0.5, label="Ballistic")
    ax_alpha.set_ylabel("Local α")
    ax_alpha.set_xlabel("Time (Minutes)")
    ax_alpha.set_ylim(0, 2.5)
    ax_alpha.grid(True)
    ax_alpha.legend(loc="right")

    # Customize main plot
    # ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Mean Square Displacement")
    ax_main.legend(loc="upper left")
    ax_main.grid(True, which="both", linestyle=":", alpha=0.5)

    # Add title
    fig.suptitle("Mean Square Displacement Analysis", y=1.02, fontsize=12)

    # plt.tight_layout()

    return fig


def plot_cell_trajectories(ground_truth: np.ndarray, predictions: np.ndarray) -> str:
    """Plots ground truth and predicted trajectories, centered on the origin."""

    num_trajectories, num_timesteps, _ = ground_truth.shape

    fig, ax = plt.subplots(figsize=(12, 10))

    # Center the trajectories
    centered_ground_truth = ground_truth - ground_truth.mean(axis=(0, 1))
    centered_predictions = predictions - predictions.mean(axis=(0, 1))

    # Plot ground truth
    for i in range(num_trajectories):
        x, y = centered_ground_truth[i, :, 0], centered_ground_truth[i, :, 1]
        ax.plot(x, y, "b-", linewidth=2, alpha=0.7, label="Ground Truth" if i == 0 else "")
        ax.plot(x[-1], y[-1], "bo", markersize=8)  # End point

    # Plot predictions
    for i in range(num_trajectories):
        x, y = centered_predictions[i, :, 0], centered_predictions[i, :, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with a color gradient
        lc = LineCollection(segments, cmap="Reds", alpha=0.7)  # type: ignore
        lc.set_array(np.linspace(0, 1, len(x)))
        line = ax.add_collection(lc)

        ax.plot(x[0], y[0], "ro", markersize=8)  # Start point

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Cell Trajectory Rollouts: Ground Truth vs Predictions")
    ax.legend()

    # Add colorbar to show prediction time
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Prediction Time")

    plt.tight_layout()

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name, format="png")
    plt.close(fig)

    return temp_file.name


def animate_graph(
    graph: Data,
    predicted_trajectories: Tensor,
    config: dict,
    name: str = "graph",
):
    """Animates dynamic graph and shows plot"""

    num_frames = config["time_window"]
    num_nodes = graph.num_nodes // num_frames

    fig, ax = plt.subplots(dpi=250)

    nodes = predicted_trajectories.view(num_frames * num_nodes, 2).cpu().numpy()  # type: ignore
    trajs = predicted_trajectories.view(num_frames, num_nodes, 2).cpu().numpy()  # type: ignore
    gt_trajs = graph.trajectories.view(num_frames, num_nodes, 2).cpu().numpy()
    nodes_gt = graph.trajectories.view(num_frames * num_nodes, 2).cpu().numpy()
    if sum(config["temporal_edges"]) > 0:
        temp_edge_index_length = graph.temporal_edge_index.shape[1]
        total_length = graph.edge_index.shape[1]  # type: ignore
        # Remove last edges (temporal)
        graph.edge_index = graph.edge_index[:, : total_length - temp_edge_index_length]  # type: ignore
    if graph.edge_index is not None:
        edge_indexes = graph.edge_index.view(2, num_frames, -1).cpu().numpy()  # type: ignore
        plot_edges = True
    else:
        plot_edges = False

    # Set up plot
    ax.set_xlim(np.nanmin(trajs), np.nanmax(trajs))
    ax.set_ylim(np.nanmin(trajs), np.nanmax(trajs))

    # Initialize lines and points
    lines = [
        ax.plot(
            trajs[0, i, 0], trajs[0, i, 1], "-", color="tab:red", linewidth=0.4, alpha=0.4
        )[0]
        for i in range(num_nodes)
    ]
    gt_lines = [
        ax.plot(
            gt_trajs[0, i, 0],
            gt_trajs[0, i, 1],
            "-",
            color="tab:blue",
            linewidth=0.4,
            alpha=0.4,
        )[0]
        for i in range(num_nodes)
    ]
    points = ax.plot(
        trajs[0, :, 0],
        trajs[0, :, 1],
        "o",
        color="tab:blue",
        ms=2.0,
        alpha=1,
        label="Predictions",
    )[0]
    gt_points = ax.plot(
        gt_trajs[0, :, 0],
        gt_trajs[0, :, 1],
        "o",
        color="tab:red",
        ms=2.0,
        alpha=1,
        label="Ground Truth",
    )[0]
    edge_lines = (
        ax.plot([], [], "-", color="gray", linewidth=0.08, alpha=0.4)[0]
        if plot_edges
        else None
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Animation update function
    def update(frame):
        if edge_lines is not None:
            start_indices = edge_indexes[0][frame]
            end_indices = edge_indexes[1][frame]

            # Extract start and end coordinates
            start_x = nodes[start_indices, 0]
            start_y = nodes[start_indices, 1]
            end_x = nodes[end_indices, 0]
            end_y = nodes[end_indices, 1]

            # Prepare data for line plot (alternate start and end points)
            x_edges = np.column_stack((start_x, end_x)).flatten()
            y_edges = np.column_stack((start_y, end_y)).flatten()

            # Update the line plot
            edge_lines.set_data(x_edges, y_edges)

        for i in range(num_nodes):
            lines[i].set_data(trajs[: frame + 1, i, 0], trajs[: frame + 1, i, 1])
            gt_lines[i].set_data(gt_trajs[: frame + 1, i, 0], gt_trajs[: frame + 1, i, 1])

        points.set_data(trajs[frame, :, 0], trajs[frame, :, 1])
        gt_points.set_data(gt_trajs[frame, :, 0], gt_trajs[frame, :, 1])

        # plt.text(0.9, 0.1, f"Frame: {frame + 1}/{num_frames}", fontsize=9)
        return tuple(
            lines + gt_lines + [points] + [gt_points] + ([edge_lines] if plot_edges else [])
        )

    ax.legend(handles=[points, gt_points], loc="upper left", fontsize=4)
    plt.tight_layout()

    fps = 3
    anim = FuncAnimation(  # Create animation
        fig,
        update,  # type: ignore
        frames=num_frames,
        interval=1000 // fps,  # type: ignore
    )
    name = f"figures/{name}.gif"
    anim.save(name, writer="pillow", fps=fps)

    return name


def animate_prior_graph(
    prior_traj: Tensor,
    config: dict,
    name: str = "prior_graph",
):
    """Animates prior dynamic graph and shows plot
    prior_traj: Tensor of shape (frames, nodes, 2)
    """

    num_frames = config["time_window"]
    num_nodes = prior_traj.shape[1]

    fig, ax = plt.subplots(dpi=250)

    trajs = prior_traj.cpu().numpy()  # type: ignore

    # Set up plot
    ax.set_xlim(np.nanmin(trajs), np.nanmax(trajs))
    ax.set_ylim(np.nanmin(trajs), np.nanmax(trajs))

    # Initialize lines and points
    lines = [
        ax.plot(
            trajs[0, i, 0], trajs[0, i, 1], "-", color="tab:orange", linewidth=0.4, alpha=0.4
        )[0]
        for i in range(num_nodes)
    ]

    points = ax.plot(
        trajs[0, :, 0],
        trajs[0, :, 1],
        "o",
        color="tab:orange",
        ms=2.0,
        alpha=1,
        label="Predictions",
    )[0]

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Animation update function
    def update(frame):
        for i in range(num_nodes):
            lines[i].set_data(trajs[: frame + 1, i, 0], trajs[: frame + 1, i, 1])

        points.set_data(trajs[frame, :, 0], trajs[frame, :, 1])

        return tuple(lines + [points])

    plt.tight_layout()

    fps = 3
    anim = FuncAnimation(  # Create animation
        fig, update, frames=num_frames, interval=1000 // fps
    )
    name = f"figures/{name}.gif"
    anim.save(name, writer="pillow", fps=fps)

    return name


def animate_cell_trajectories(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    pred_graphs,
    config: dict,
    debug: bool = False,
):
    """Animates ground truth and predicted trajectories"""

    num_samples, num_timesteps, num_trajectories, _ = ground_truth.shape

    fig, ax = plt.subplots(figsize=(12, 10))

    # Set up plot
    ax.set_xlim(np.nanmin(ground_truth), np.nanmax(ground_truth))
    ax.set_ylim(np.nanmin(ground_truth), np.nanmax(ground_truth))
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Initialize lines and points
    truth_lines = [
        ax.plot([], [], "-", color="tab:blue", linewidth=0.6, alpha=0.6)[0]
        for i in range(num_trajectories)
    ]
    pred_lines = [
        ax.plot([], [], "-", color="tab:red", linewidth=0.6, alpha=0.6)[0]
        for i in range(num_trajectories)
    ]
    edge_lines = ax.plot([], [], "-", color="gray", linewidth=0.10, alpha=0.5)[0]
    truth_points = ax.plot(
        [], [], "o", color="tab:blue", markersize=5, alpha=1.0, label="Ground truth"
    )[0]
    pred_points = ax.plot(
        [], [], "o", color="tab:red", markersize=5, alpha=1.0, label="Predictions"
    )[0]

    # Animation update function
    def update(frame):
        t = frame % num_timesteps
        n = frame // num_timesteps

        # Clear previous lines and points
        for i in range(num_trajectories):
            truth_lines[i].set_data([], [])
            pred_lines[i].set_data([], [])
        truth_points.set_data([], [])
        pred_points.set_data([], [])
        edge_lines.set_data([], [])

        for i in range(num_trajectories):
            pred_lines[i].set_data(
                predictions[n, : t + 1, i, 0], predictions[n, : t + 1, i, 1]
            )
            truth_lines[i].set_data(
                ground_truth[n, : t + 1, i, 0], ground_truth[n, : t + 1, i, 1]
            )
        truth_points.set_data(ground_truth[n, t, :, 0], ground_truth[n, t, :, 1])
        pred_points.set_data(predictions[n, t, :, 0], predictions[n, t, :, 1])

        if config["plot_edges"]:
            edge_indices = pred_graphs[n][t].edge_index

            start_indices = edge_indices[0]
            end_indices = edge_indices[1]

            # Extract start and end coordinates
            start_x = pred_graphs[n][t].pos[start_indices, 0]
            start_y = pred_graphs[n][t].pos[start_indices, 1]
            end_x = pred_graphs[n][t].pos[end_indices, 0]
            end_y = pred_graphs[n][t].pos[end_indices, 1]

            # Prepare data for line plot (alternate start and end points)
            x_edges = np.column_stack((start_x, end_x)).flatten()
            y_edges = np.column_stack((start_y, end_y)).flatten()

            # Update the line plot
            edge_lines.set_data(x_edges, y_edges)

        ax.set_title(
            f"Cell Trajectory Rollouts (Frame {t + 1}/{num_timesteps} of Sample {n + 1}/{num_samples})"
        )
        return truth_lines + pred_lines + [truth_points] + [pred_points]

    fps = 2

    ax.legend(handles=[truth_points, pred_points], loc="upper left")

    # Create animation
    anim = FuncAnimation(
        fig,
        func=update,
        frames=int(num_samples * num_timesteps),
        interval=1000 // fps,
        repeat=False,
    )

    plt.tight_layout()

    if debug:
        # anim.save("temp.gif", writer="pillow", fps=fps)
        # plt.close(fig)
        return None
    else:
        # Save the animation to a temporary file
        # temp_file = tempfile.NamedTemporaryFile()
        # writer = PillowWriter(fps=fps)
        anim.save("temp.gif", writer="imagemagick", fps=fps)
        return "temp.gif"


def animate_3d_trajectories(ground_truth: np.ndarray, predictions: np.ndarray, config: dict):
    """Animates 2D trajectories over time in 3D space.
    Args:
        ground_truth: ndarray of shape (n_trajectories, frames, 2)
        predictions: ndarray of shape (n_trajectories, frames, 2)
    """
    num_trajectories, num_timesteps, _ = ground_truth.shape

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set up plot
    ax.set(xlim3d=(np.nanmin(ground_truth), np.nanmax(ground_truth)), xlabel="X")
    ax.set(ylim3d=(0, num_timesteps), ylabel="Time")
    ax.set(zlim3d=(np.nanmin(ground_truth), np.nanmax(ground_truth)), zlabel="Y")

    # Initialize lines and points
    lines = [
        ax.plot([], [], [], ":", color="tab:blue", linewidth=1, alpha=0.5)[0]
        for i in range(num_trajectories)
    ]
    lines_gt = [
        ax.plot([], [], [], ":", color="tab:red", linewidth=1, alpha=0.5)[0]
        for i in range(num_trajectories)
    ]
    points = ax.plot([], [], [], "o", color="tab:blue", markersize=4, alpha=0.9)[0]
    points_gt = ax.plot([], [], [], "o", color="tab:red", markersize=4, alpha=0.9)[0]

    # Animation update function
    def update(frame):
        for i in range(num_trajectories):
            data = (
                predictions[i, : frame + 1, 0],
                np.arange(frame + 1),
                predictions[i, : frame + 1, 1],
            )
            data_gt = (
                ground_truth[i, : frame + 1, 0],
                np.arange(frame + 1),
                ground_truth[i, : frame + 1, 1],
            )
            lines[i].set_data_3d(data)  # type: ignore
            lines_gt[i].set_data_3d(data_gt)  # type: ignore

        points.set_data_3d(  # type: ignore
            predictions[:, frame, 0],
            np.repeat(frame, num_trajectories),
            predictions[:, frame, 1],
        )
        points_gt.set_data_3d(  # type: ignore
            ground_truth[:, frame, 0],
            np.repeat(frame, num_trajectories),
            ground_truth[:, frame, 1],
        )

        return tuple(lines + lines_gt + [points] + [points_gt])

    fps = 3

    # Create animation
    anim = FuncAnimation(
        fig,
        func=update,
        frames=num_timesteps,
        interval=1000 // fps,
    )

    name = "figures/3d_trajectories.gif"
    anim.save(name, writer="pillow", fps=fps)
    return name


def plot_final_displacement_error(ground_truth: np.ndarray, predictions: np.ndarray):
    """Plot final displacement error for each trajectory."""

    final_errors = final_displacement_error(ground_truth, predictions)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(final_errors)), final_errors)
    ax.set_xlabel("Trajectory")
    ax.set_ylabel("Final Displacement Error")
    ax.set_title("Final Displacement Error for Each Trajectory")
    ax.set_xticks(range(len(final_errors)))
    ax.grid(True, axis="y")
    plt.tight_layout()

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name, format="png")
    plt.close(fig)

    return temp_file.name


def plot_error_cdf(ground_truth: np.ndarray, predictions: np.ndarray):
    errors = np.mean(np.linalg.norm(ground_truth - predictions, axis=3), axis=0).flatten()
    sorted_errors = np.sort(errors)
    cumulative = np.arange(len(sorted_errors)) / float(len(sorted_errors))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_errors, cumulative, "b-")
    ax.set_xlabel("Displacement Error")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution of Displacement Errors")
    ax.grid(True)
    plt.tight_layout()

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name, format="png")
    plt.close(fig)

    return temp_file.name


def plot_centered_trajectories(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    num_trajectories: int = 50,
    num_plots: int = 2,
    size: int = 10,
    debugging: bool = False,
):
    fig, axs = plt.subplots(num_plots, 2, figsize=(12, 6 * num_plots))
    if axs.size == 2:
        axs = axs.reshape(1, 2)

    num_plots = min(num_plots, len(ground_truth))
    num_trajectories = min(num_trajectories, min([rec.shape[0] for rec in ground_truth]))
    indexes = np.random.choice(len(ground_truth), num_plots, replace=False)
    ground_truth, predictions = ground_truth[indexes], predictions[indexes]

    for i in range(num_plots):
        num_cells, num_steps, _ = ground_truth[i].shape
        cell_ids_to_plot = np.random.choice(num_cells, num_trajectories, replace=False)

        for cell_id in cell_ids_to_plot:
            # Extract trajectories for the current cell
            gt_trajectory = ground_truth[i, cell_id, :, :]
            pred_trajectory = predictions[i, cell_id, :, :]

            # Center the trajectories at (0,0)
            gt_centered = gt_trajectory - gt_trajectory[0]
            pred_centered = pred_trajectory - pred_trajectory[0]

            # Plot ground truth
            axs[i, 0].plot(gt_centered[:, 0], gt_centered[:, 1], alpha=0.7, linewidth=1.5)

            # Plot prediction
            axs[i, 1].plot(pred_centered[:, 0], pred_centered[:, 1], alpha=0.7, linewidth=1.5)

        axs[i, 0].set_xlim(-size, size)
        axs[i, 1].set_xlim(-size, size)
        axs[i, 0].set_ylim(-size, size)
        axs[i, 1].set_ylim(-size, size)
        axs[i, 0].grid(True)
        axs[i, 1].grid(True)
        axs[i, 0].set_ylabel("Y")
        axs[i, 1].set_ylabel("Y")
        axs[i, 0].set_xlabel("X")
        axs[i, 1].set_xlabel("X")

    axs[0, 0].set_title("Ground Truth Trajectories")
    axs[0, 1].set_title("Predicted Trajectories")

    plt.tight_layout()

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name, format="png")
    plt.close(fig)

    return temp_file.name


def plot_trajectories(trajectories: Tensor):
    """
    Plot the evolution of trajectories

    Args:
        trajectories: Tensor of shape (frames, nodes, 2) containing trajectories
    """
    n_frames, n_nodes, _ = trajectories.shape

    trajs = trajectories.cpu().numpy()

    # Create figure with three subplots
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all trajectories
    ax.set_title("Prior Trajectories (x0)")
    for i in range(n_nodes):
        traj = trajs[:, i]
        ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=2, alpha=0.7)
        ax.scatter(traj[0, 0], traj[0, 1], c="lime", marker="o", s=50, zorder=5)  # Start
        ax.scatter(traj[-1, 0], traj[-1, 1], c="red", marker="x", s=50, zorder=5)  # End

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    # plt.show(block=False)


def plot_trajectory_evolution(
    x0_traj: Tensor,
    x1_traj: Tensor,
    gt_traj: Tensor,
    fig_size=(15, 6),
    max_trajectories=5,  # Limit number of trajectories to avoid cluttering
    cond_length: int = 0,
    max_x_and_y: float = 1.0,
):
    """
    Plot the evolution of trajectories from x0 (prior) to x1 (target) through the vector field.

    Args:
        x0_traj: Tensor of shape (batch_size, frames*2) containing prior trajectories
        x1_traj: Tensor of shape (batch_size, frames*2) containing predicted trajectories
        gt_traj: Tensor of shape (batch_size, frames*2) containing ground truth trajectories
        fig_size: Size of the figure (width, height)
        max_trajectories: Maximum number of trajectories to plot
    """
    batch_size = x0_traj.shape[0]
    n_frames = x0_traj.shape[1]
    c = cond_length

    if x1_traj.shape[-1] == 3:
        x0_traj = x0_traj[:, :, :2]
        x1_traj = x1_traj[:, :, :2]
        gt_traj = gt_traj[:, :, :2]

    # Reshape trajectories to (batch, frames, 2)
    x0_traj = x0_traj.reshape(batch_size, n_frames, 2)
    x1_traj = x1_traj.reshape(batch_size, n_frames, 2)
    gt_traj = gt_traj.reshape(batch_size, n_frames, 2)

    # Select subset of trajectories if needed
    if batch_size > max_trajectories:
        indices = torch.randperm(batch_size)[:max_trajectories]
        x0_traj = x0_traj[indices]
        x1_traj = x1_traj[indices]
        gt_traj = gt_traj[indices]
        batch_size = max_trajectories

    # Create figure with three subplots
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # Prior trajectories
    ax2 = fig.add_subplot(gs[0, 1])  # Target trajectories
    ax3 = fig.add_subplot(gs[0, 2])  # Ground truth trajectories
    plt.style.use("seaborn-v0_8-paper")

    # Plot prior trajectories
    ax1.set_title("Prior Trajectories (x0)")
    for i in range(batch_size):
        traj = x0_traj[i].cpu().numpy()
        ax1.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, alpha=0.7)
        ax1.scatter(traj[0, 0], traj[0, 1], c="lime", marker="o", s=50, zorder=5)  # Start
        if c > 0:
            ax1.plot(traj[:c, 0], traj[:c, 1], "-o", c="lime", markersize=3)  # Conditioning
        ax1.scatter(traj[-1, 0], traj[-1, 1], c="red", marker="x", s=50, zorder=5)  # End

    # Plot predicted trajectories
    ax2.set_title("Predicted Trajectories (x1)")
    for i in range(batch_size):
        traj = x1_traj[i].cpu().numpy()
        ax2.plot(traj[c:, 0], traj[c:, 1], "-o", markersize=3, alpha=0.7)
        ax2.scatter(traj[0, 0], traj[0, 1], c="lime", marker="o", s=50, zorder=5)  # Start
        if c > 0:
            ax2.plot(traj[1:c, 0], traj[1:c, 1], "-o", c="lime", markersize=3)  # Conditioning
        ax2.scatter(traj[-1, 0], traj[-1, 1], c="red", marker="x", s=50, zorder=5)  # End

    # Plot ground truth trajectories
    ax3.set_title("Ground truth Trajectories")
    for i in range(batch_size):
        traj = gt_traj[i].cpu().numpy()
        ax3.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, alpha=0.7)
        ax3.scatter(traj[0, 0], traj[0, 1], c="lime", marker="o", s=50, zorder=5)  # Start
        if c > 0:
            ax3.plot(traj[1:c, 0], traj[1:c, 1], "-o", c="lime", markersize=3)  # Conditioning
        ax3.scatter(traj[-1, 0], traj[-1, 1], c="red", marker="x", s=50, zorder=5)  # End

    ax3.set_aspect("equal")
    x_min, x_max = ax3.get_xlim()
    y_min, y_max = ax3.get_ylim()
    max_range = max(x_max - x_min, y_max - y_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    # Set common properties for all axes
    for ax in [ax1, ax2, ax3]:
        # Set equal aspect ratio
        ax.set_aspect("equal")
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.grid(True, alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    return fig


def plot_flows(
    priors, ground_truths, predictions, model, config, debugging: bool
) -> plt.Figure:
    w = 1
    # points_small = 20j
    # points_real_small = 20
    # Y_small, X_small = np.mgrid[-w:w:points_small, -w:w:points_small]
    # gridpoints_small = torch.tensor(np.stack([X_small.flatten(), Y_small.flatten()], axis=1)).type(
    #     torch.float32
    # )
    priors = priors[0].view(1, -1)
    input_frames = config["time_window"]
    output_frames = config["time_window"]
    cond_frames = config["cond_frames"]
    cond = priors[:, : cond_frames * 2]

    with torch.no_grad():
        trajectories = odeint(
            func=lambda t, x: model.forward(t, x, cond),
            y0=priors[:, cond_frames * 2 :],
            t=torch.linspace(0, 1, 101, device=priors.device),
            method="dopri5",
        )

    ts = torch.linspace(0, 1, 101).to(priors.device)
    for i, t in enumerate(ts):
        fig, axis = plt.subplots(1, 3, figsize=(6 * 3, 6 * 1))
        # model = models[name]
        # cnf = DEFunc(CNF(model))
        # nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
        # cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
        # with torch.no_grad():
        #     if t > 0:
        #         aug_traj = (
        #             cnf_model[1]
        #             .to(device)
        #             .trajectory(
        #                 Augmenter(1, 1)(gridpoints).to(device),
        #                 t_span=torch.linspace(t, 0, 201).to(device),
        #             )
        #         )[-1].cpu()
        #         log_probs = log_8gaussian_density(aug_traj[:, 1:]) - aug_traj[:, 0]
        #     else:
        #         log_probs = log_8gaussian_density(gridpoints)
        # log_probs = log_probs.reshape(Y.shape)
        # ax = axis[0]
        # ax.pcolormesh(X, Y, torch.exp(log_probs), vmax=1)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlim(-w, w)
        # ax.set_ylim(-w, w)
        # ax.set_title(f"{name}", fontsize=30)

        # Plot prior trajectories
        ax = axis[0]
        ax.set_title("Prior Trajectories (x0)")
        for k in range(priors.shape[0]):
            traj = priors[k].view(input_frames, 2).cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=2, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c="lime", marker="o", s=50, zorder=5)  # Start
            if cond_frames > 0:
                ax.plot(
                    traj[:cond_frames, 0], traj[:cond_frames, 1], "-o", c="lime", markersize=2
                )  # Conditioning
            ax.scatter(traj[-1, 0], traj[-1, 1], c="red", marker="x", s=50, zorder=5)  # End

        # Quiver plot
        with torch.no_grad():
            v_t = (
                model(
                    t,
                    priors[:, cond_frames * 2 :],
                    cond,
                )
                .view(-1, output_frames, 2)
                .cpu()
                .numpy()
            )
        X = (
            priors[:, cond_frames * 2 :]
            .view(-1, output_frames, 2)[:, :, 0]
            .cpu()
            .detach()
            .numpy()
        )
        Y = (
            priors[:, cond_frames * 2 :]
            .view(-1, output_frames, 2)[:, :, 1]
            .cpu()
            .detach()
            .numpy()
        )
        ax = axis[1]
        ax.set_title("Velocity Field (vt)")
        ax.quiver(
            X,
            Y,
            v_t[:, :, 0],
            v_t[:, :, 1],
            np.sqrt(np.sum(v_t**2, axis=-1)),
            cmap="coolwarm",
            # scale=50.0,
            # width=0.015,
            # pivot="mid",
        )

        ax = axis[2]
        ax.set_title("Probability Paths")
        sample_traj = trajectories.view(-1, priors.shape[0], output_frames, 2).cpu().numpy()  # type: ignore
        for j in range(priors.shape[0]):
            ax.scatter(
                sample_traj[0, j, :, 0], sample_traj[0, j, :, 1], s=10, alpha=0.8, c="black"
            )
            debug = sample_traj[:i, j, :, 0]
            debug = sample_traj[:i, j, :, 1]
            ax.scatter(
                sample_traj[:i, j, :, 0], sample_traj[:i, j, :, 1], s=0.2, alpha=0.2, c="olive"
            )
            ax.scatter(
                sample_traj[i, j, :, 0], sample_traj[i, j, :, 1], s=4, alpha=1, c="blue"
            )

        for ax in axis:
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlim(-w, w)
            # ax.set_ylim(-w, w)
            # ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")

        plt.suptitle(f"Optimal Transport Conditional Flow Matching: T={t:0.2f}", fontsize=14)
        os.makedirs("figures/", exist_ok=True)
        plt.savefig(f"figures/{t:0.2f}.png", dpi=200)
        plt.close(fig)

    gif_name = "flows"
    with imageio.get_writer(f"{gif_name}.gif", mode="I") as writer:  # type: ignore
        for filename in [f"figures/{t:0.2f}.png" for t in ts] + [
            f"figures/{ts[-1].item():0.2f}.png"
        ] * 10:
            image = imageio.imread(filename)
            writer.append_data(image)  # type: ignore

    return fig


def plot_neighbour_dist_over_time(means, stds, means_gt, stds_gt, means_x0) -> plt.Figure:
    """Plots the mean distances to the nearest cells neighbours over time."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean distances
    ax.plot(
        range(len(means)),
        means,
        "-o",
        markersize=2.5,
        linewidth=1.5,
        label="Predictions",
        color="tab:blue",
    )
    ax.fill_between(
        range(len(means)),
        means - stds,
        means + stds,
        alpha=0.2,
        color="tab:blue",
    )

    # Plot ground truth mean distances
    ax.plot(
        range(len(means_gt)),
        means_gt,
        "-o",
        markersize=2.5,
        linewidth=1.5,
        label="Ground Truth",
        color="tab:red",
    )
    ax.fill_between(
        range(len(means_gt)),
        means_gt - stds_gt,
        means_gt + stds_gt,
        alpha=0.2,
        color="tab:red",
    )

    # Plot mean distances for prior
    ax.plot(
        range(len(means_x0)),
        means_x0,
        "-o",
        markersize=2.5,
        linewidth=1.5,
        label="Prior",
        color="tab:orange",
    )

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Mean distance to neighbours")
    ax.set_title("Mean Distance to K-Nearest Neighbours Over Time")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig


def visualize_flow_3d(prior_data, predicted_data, range=(3, 3), fps=30, duration=5):
    """
    Creates an animated 3D visualization of flow matching results using matplotlib's
    object-oriented interface.

    Parameters:
    -----------
    prior_data : np.ndarray
        Array of shape (n_trajs, frames, 2 or 3) containing the prior distribution points
    predicted_data : np.ndarray
        Array of shape (n_trajs, frames, 2 or 3) containing the predicted distribution points
    fps : int
        Frames per second for the animation
    duration : float
        Duration of the animation in seconds

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the animation
    anim : matplotlib.animation.FuncAnimation
        The animation object that can be saved or displayed
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Extract dimensions from the data
    n_trajs = prior_data.shape[0]
    polar = prior_data.shape[-1] == 3

    # If 2d data, add a third dimension with time
    if not polar:
        time = np.arange(prior_data.shape[1]).reshape(1, -1, 1).repeat(n_trajs, axis=0)
        prior_data = np.concatenate([prior_data, time], axis=-1)
        predicted_data = np.concatenate([predicted_data, time], axis=-1)

    # Calculate the axis limits to keep them fixed throughout the animation
    all_data = np.concatenate([prior_data, predicted_data], axis=0)
    x_min, x_max = -range[0], range[0]  # all_data[..., 0].min(), all_data[..., 0].max()
    y_min, y_max = -range[1], range[1]  # all_data[..., 1].min(), all_data[..., 1].max()
    z_min, z_max = all_data[..., 2].min(), all_data[..., 2].max()

    # Add some padding to the limits
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])  # type: ignore
    ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])  # type: ignore
    ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])  # type: ignore

    # Set labels and title
    ax.set_xlabel("Velocity Magnitude" if polar else "X Velocity")
    ax.set_ylabel("sin(Angular Velocity)" if polar else "Y Velocity")
    ax.set_zlabel("cos(Angular Velocity)" if polar else "Time")  # type: ignore
    ax.set_title("Flow Visualization")

    # Initialize scatter plot with the prior data
    scatter_x0 = ax.scatter(
        prior_data[..., 0].flatten(),
        prior_data[..., 1].flatten(),
        prior_data[..., 2].flatten(),
        c="tab:orange",
        alpha=0.4,
        marker=".",
        # marker_size=1,
    )
    scatter_xt = ax.scatter(
        prior_data[..., 0].flatten(),
        prior_data[..., 1].flatten(),
        prior_data[..., 2].flatten(),
        c="tab:blue",
        alpha=0.6,
        marker=".",
        # marker_size=1,
    )
    # paths = ax.plot([], [], [], ":", color="gray", alpha=0.6, linewidth=0.5)[0]

    # Calculate number of frames
    n_frames = int(fps * duration)

    def update(frame):
        """Update function for animation"""
        # Calculate interpolation factor (0 to 1)
        t = frame / (n_frames - 1)

        # Interpolate between prior and predicted positions
        current_points = (1 - t) * prior_data + t * predicted_data

        # Update scatter plot positions
        scatter_xt._offsets3d = (  # type: ignore
            current_points[..., 0].flatten(),
            current_points[..., 1].flatten(),
            current_points[..., 2].flatten(),
        )

        # Update paths from prior to current points
        # paths.set_data_3d(
        #     np.stack([prior_data[..., 0], current_points[..., 0]], axis=0).flatten(),
        #     np.stack([prior_data[..., 1], current_points[..., 1]], axis=0).flatten(),
        #     np.stack([prior_data[..., 2], current_points[..., 2]], axis=0).flatten(),
        # )

        # Update title to show progress
        ax.set_title(f"Flow Matching Visualization (t={t:.2f})")

        return (
            scatter_x0,
            scatter_xt,
            # paths,
        )

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,  # interval in milliseconds
        blit=False,
    )  # blit=False ensures the title updates

    # Adjust the layout to prevent clipping
    plt.tight_layout()

    anim.save("figures/flows_3d.gif", writer="pillow", fps=fps)

    # plt.show()

    return "figures/flows.gif"


def animate_flows(ground_truth: np.ndarray, predictions: np.ndarray, debugging: bool) -> str:
    """
    Create a side-by-side animation comparing ground truth and predicted trajectories.

    Parameters:
    -----------
    ground_truth : numpy.ndarray
        Ground truth trajectories of shape (n_trajectories, frames, 2)
    predictions : numpy.ndarray
        Predicted trajectories of shape (time_steps, n_trajectories, frames, 2)
    config: dict

    Returns:
    --------
    save_path : str
        Path to the saved GIF file
    """
    # Extract dimensions from the data
    time_steps, n_trajectories, frames, _ = predictions.shape

    # Calculate linear probability paths for ground truth
    ground_truth = np.linspace(predictions[0], ground_truth, time_steps)

    # Create a figure with two subplots side by side
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(15, 7), dpi=200)
    fig.suptitle("Trajectory Flow Evolution (t=0 to t=1)", fontsize=14)

    # Set up the subplots
    ax1.set_title("Ground Truth (linear interpolation)")
    ax2.set_title("Predictions")

    # Generate distinct colors for each trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, n_trajectories))  # type: ignore

    # Initialize empty line and scatter objects for both plots
    gt_lines = []
    pred_lines = []
    gt_dots = []
    pred_dots = []

    # Create initial empty lines and dots for each trajectory
    for i in range(n_trajectories):
        # Ground truth
        (line_gt,) = ax1.plot([], [], "-", color=colors[i], alpha=0.4, linewidth=2)
        dot_gt = ax1.scatter([], [], color=colors[i], s=20)
        gt_lines.append(line_gt)
        gt_dots.append(dot_gt)

        # Predictions
        (line_pred,) = ax2.plot([], [], "-", color=colors[i], alpha=0.4, linewidth=2)
        dot_pred = ax2.scatter([], [], color=colors[i], s=20)
        pred_lines.append(line_pred)
        pred_dots.append(dot_pred)

    # Find data bounds for consistent axes
    all_data = np.concatenate([ground_truth, predictions])
    x_min, x_max = all_data[..., 0].min(), all_data[..., 0].max()
    y_min, y_max = all_data[..., 1].min(), all_data[..., 1].max()

    # Add some padding to the bounds
    padding = 0.1 * max(x_max - x_min, y_max - y_min)

    # Set consistent axes limits
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Add a time indicator text
    time_text = fig.text(0.5, 0.02, "", ha="center")

    def init():
        for line in gt_lines + pred_lines:
            line.set_data([], [])
        for dot in gt_dots + pred_dots:
            dot.set_offsets(np.c_[[], []])
        time_text.set_text("")
        return gt_lines + pred_lines + gt_dots + pred_dots + [time_text]

    def update(frame):
        # Update time indicator
        time_text.set_text(f"t = {frame / time_steps:.2f}")

        # Update each trajectory
        for i in range(n_trajectories):
            # Ground truth
            gt_data = ground_truth[frame, i]
            gt_lines[i].set_data(gt_data[:, 0], gt_data[:, 1])
            gt_dots[i].set_offsets(gt_data)

            # Predictions
            pred_data = predictions[frame, i]
            pred_lines[i].set_data(pred_data[:, 0], pred_data[:, 1])
            pred_dots[i].set_offsets(pred_data)

        return gt_lines + pred_lines + gt_dots + pred_dots + [time_text]

    frames = list(range(time_steps)) + [time_steps - 1] * int((0.4 * time_steps))

    # Create the animation
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,  # type: ignore
        init_func=init,
        blit=False,
        interval=1000 / 15,
    )

    save_path = "figures/flows_2d.gif"

    plt.tight_layout()

    # Save the animation if a path is provided
    anim.save(save_path, writer="pillow", fps=30)

    # if debugging:
    #     plt.show()
    # else:
    plt.close()
    return save_path


def plot_3d_trajectories(
    trajectories,
    gt,
    c=0,
    figsize=(10, 5),
    save_path="trajectories_3d.svg",
):
    """
    Plots 3D trajectories with customized appearance.

    Parameters:
    -----------
    trajectories : numpy.ndarray
        Array of shape (n, t, 3) where:
        - n is the number of trajectories
        - t is the number of time steps
        - 3 is the dimensionality (x, y, z)
    c : int, default=0
        Number of observed frames at the beginning of each trajectory to color black
    figsize : tuple, default=(10, 8)
        Figure size as (width, height) in inches
    save_path : str, default='trajectories_3d.svg'
        Path to save the output SVG file
    linewidth : float, default=1.5
        Width of the trajectory lines
    marker_size : float, default=30
        Size of regular markers
    last_marker_size : float, default=80
        Size of the last marker in each trajectory
    axis_linewidth : float, default=2.0
        Width of the axis lines

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)

    # Create 3D axes
    grid = plt.GridSpec(1, 2, figure=fig)
    ax_pred = fig.add_subplot(grid[0, 0], projection="3d")
    ax_gt = fig.add_subplot(grid[0, 1], projection="3d")

    # Get nice color cycle - using a colorblind-friendly palette
    colors = plt.cm.tab10.colors  # type: ignore

    axis_linewidth = 2.0
    linewidth = 1.5
    marker_size = 30
    last_marker_size = 60

    # Store min/max values to set axis limits later
    all_points = trajectories.reshape(-1, 3)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    for a, ax in enumerate([ax_pred, ax_gt]):
        # Set thick axes
        ax.xaxis.line.set_linewidth(axis_linewidth)  # type: ignore
        ax.yaxis.line.set_linewidth(axis_linewidth)  # type: ignore
        ax.zaxis.line.set_linewidth(axis_linewidth)  # type: ignore

        # Make ticks thicker
        ax.tick_params(axis="x", width=axis_linewidth / 1.5)
        ax.tick_params(axis="y", width=axis_linewidth / 1.5)
        ax.tick_params(axis="z", width=axis_linewidth / 1.5)  # type: ignore

        trajs = trajectories if a == 0 else gt
        # Plot each trajectory
        for i, traj in enumerate(trajs):
            color = colors[i % len(colors)]

            # Plot observed frames (black)
            if c > 0:
                observed = traj[:c]
                # Plot line segments
                ax.plot(
                    observed[:, 0],
                    observed[:, 1],
                    observed[:, 2],
                    color="black",
                    linewidth=linewidth / 2,
                    zorder=3,
                    alpha=0.6,
                )

                # Plot dots
                ax.scatter(
                    observed[:, 0],
                    observed[:, 1],
                    observed[:, 2],
                    color="black",
                    s=marker_size / 2,  # type: ignore
                    zorder=4,
                    alpha=0.6,
                )

            # Plot future frames (colored)
            future = traj[c:]
            if len(future) > 0:
                # Plot line segments
                ax.plot(
                    future[:, 0],
                    future[:, 1],
                    future[:, 2],
                    color=color,
                    linewidth=linewidth,
                    zorder=1,
                    alpha=0.8,
                )

                # Plot dots (except last one)
                if len(future) > 1:
                    ax.scatter(
                        future[:-1, 0],
                        future[:-1, 1],
                        future[:-1, 2],
                        color=color,
                        s=marker_size,  # type: ignore
                        zorder=2,
                        alpha=0.85,
                    )

                # Plot last point with bigger marker
                ax.scatter(
                    future[-1:, 0],
                    future[-1:, 1],
                    future[-1:, 2],
                    color=color,
                    s=last_marker_size,  # type: ignore
                    zorder=5,
                    alpha=0.9,
                    # marker="o",
                    # edgecolors="black",
                    # linewidth=1,
                )

        # Add some padding to the axis limits for better visibility
        padding = 0.05 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])  # type: ignore

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore

        # Add a grid
        ax.grid(True, alpha=0.3)

        ax.set_title("Ground truth" if a == 1 else "Predictions", fontsize=14)

    # Improve spacing
    plt.tight_layout(w_pad=3)

    # plt.show()

    # Save as SVG
    plt.savefig(save_path, bbox_inches="tight")

    return fig, ax


def visualize_trajectories(
    ground_truth,
    predictions,
    batch_indices,
    num_samples=6,
    history_frames=8,
    prediction_frames=12,
    figsize=(16, 8),
):
    """
    Visualize trajectory predictions compared to ground truth.

    Parameters:
    -----------
    ground_truth : Tensor
        Ground truth trajectories with shape (#batch * #trajs * #frames, 2)
    predictions : list[Tensor]
        List of Predicted trajectories with shape (#trajs, #frames, 2)
    batch_indices : numpy.ndarray
        Indices indicating which trajectory belongs to which sample
    num_samples : int
        Number of samples to visualize (default: 5)
    history_frames : int
        Number of history frames provided to the model (default: 8)
    prediction_frames : int
        Number of prediction frames (default: 12)
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Set the style
    plt.style.use("seaborn-v0_8-paper")

    if ground_truth.shape[-1] == 3:
        ground_truth = ground_truth[:, :2]
        for i in range(len(predictions)):
            predictions[i] = predictions[i][:, :, :2]

    # Create colormap for ADE visualization (green to red)
    cmap = LinearSegmentedColormap.from_list("ADE", ["green", "yellow", "red"])

    ground_truth = ground_truth.cpu().numpy()
    batch_indices = batch_indices.cpu().numpy()

    # Get unique batch indices and select the first num_samples
    unique_batches = np.unique(batch_indices)
    selected_batches = unique_batches[: min(num_samples, len(unique_batches))]
    window = history_frames + prediction_frames

    # Create the figure and axes
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)

    # Prepare data for metrics table
    # metrics_data = []
    all_ades = []

    # For each selected batch
    for i, batch_idx in enumerate(selected_batches):
        ax = fig.add_subplot(gs[i])

        # Get all trajectories for this batch
        batch_mask = batch_indices == batch_idx

        batch_gt = ground_truth[batch_mask].reshape(window, -1, 2).transpose(1, 0, 2)
        batch_pred = predictions[i].cpu().numpy()

        # For each trajectory in this batch
        for j in range(len(batch_gt)):
            gt_traj = batch_gt[j]
            pred_traj = batch_pred[j]

            # Split into history and future
            gt_history = gt_traj[:history_frames]
            gt_future = gt_traj[history_frames:]
            pred_future = pred_traj[history_frames:]

            # Plot history trajectory (gray)
            ax.plot(
                gt_history[:, 0],
                gt_history[:, 1],
                "o-",
                color="tab:gray",
                linewidth=2,
                markersize=4,
                alpha=0.6,
                label="History" if j == 0 else "",
            )

            # Plot ground truth future trajectory (blue)
            ax.plot(
                gt_future[:, 0],
                gt_future[:, 1],
                "o-",
                color="tab:blue",
                linewidth=2,
                markersize=4,
                label="Ground Truth" if j == 0 else "",
                alpha=0.8,
            )

            # Create colored segments for prediction based on ADE
            points = pred_future.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Calculate ADE for each predicted point
            ade = np.sqrt(np.sum((pred_future - gt_future) ** 2, axis=1))
            all_ades.append(ade)
            ade_segments = (ade[:-1] + ade[1:]) / 2
            fde = np.sqrt(np.sum((pred_future[-1] - gt_future[-1]) ** 2))

            # Normalize ADE values for color mapping
            norm = plt.Normalize(0, max(1.0, np.max(all_ades)))

            # Create colored line collection
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)  # type: ignore
            lc.set_array(ade_segments)
            line = ax.add_collection(lc)

            # Plot prediction points
            scatter = ax.scatter(
                pred_future[:, 0],
                pred_future[:, 1],
                c=ade,
                cmap=cmap,
                norm=norm,
                s=25,
                # edgecolor="black",
                linewidth=0.5,
                label="Prediction" if j == 0 else "",
            )

        ax.set_aspect("equal")

        # Set title and labels
        ax.set_title(
            f"Sample {batch_idx}   (ADE {ade.mean():.2f}, FDE {fde.mean():.2f})", fontsize=10
        )
        ax.set_xlabel("X position", fontsize=8)
        ax.set_ylabel("Y position", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add legend (only for the first trajectory in each batch)
        if i == 0:
            ax.legend(loc="upper right")

    # Set equal aspect ratio
    ranges = []
    for ax in fig.axes:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        max_range = max(x_max - x_min, y_max - y_min) / 2
        ranges.append((max_range, x_min, x_max, y_min, y_max))
    max_range, x_min, x_max, y_min, y_max = max(ranges, key=lambda x: x[0])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    for ax in fig.axes:
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)

    # Add a colorbar for ADE
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # type: ignore
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Displacement Error")

    plt.tight_layout(rect=[0, 0, 0.94, 1])  # type: ignore
    plt.subplots_adjust(hspace=0.4)

    return fig


def visualize_trajectories_paper(
    ground_truth,
    predictions,
    batch_indices,
    num_samples=3,
    history_frames=8,
    prediction_frames=12,
    figsize=(10, 3),
):
    """
    Visualize trajectory predictions compared to ground truth.

    Parameters:
    -----------
    ground_truth : Tensor
        Ground truth trajectories with shape (#batch * #trajs * #frames, 2)
    predictions : list[Tensor]
        List of Predicted trajectories with shape (#trajs, #frames, 2)
    batch_indices : numpy.ndarray
        Indices indicating which trajectory belongs to which sample
    num_samples : int
        Number of samples to visualize (default: 5)
    history_frames : int
        Number of history frames provided to the model (default: 8)
    prediction_frames : int
        Number of prediction frames (default: 12)
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Set the style
    plt.style.use("seaborn-v0_8-paper")

    if ground_truth.shape[-1] == 3:
        ground_truth = ground_truth[:, :2]
        for i in range(len(predictions)):
            predictions[i] = predictions[i][:, :, :2]

    # Create colormap for ADE visualization (green to red)
    cmap = LinearSegmentedColormap.from_list("ADE", ["tab:green", "wheat", "tab:red"])

    ground_truth = ground_truth.cpu().numpy()
    batch_indices = batch_indices.cpu().numpy()

    # Get unique batch indices and select the first num_samples
    unique_batches = np.unique(batch_indices)

    window = history_frames + prediction_frames

    selected_batches = []
    distance = 0
    while len(selected_batches) < num_samples:
        batch = np.random.choice(unique_batches, 1, replace=False)
        mask = batch_indices == batch
        gt = ground_truth[mask].reshape(window, -1, 2)
        distance = np.linalg.norm(gt[-1] - gt[0]).mean()
        if distance > 5:
            selected_batches.append(batch[0])

    # Create the figure and axes
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig)

    # Prepare data for metrics table
    # metrics_data = []
    all_ades = []
    ades = []
    fdes = []
    max_ade = 0
    for i, batch_idx in enumerate(selected_batches):
        # Get all trajectories for this batch
        batch_mask = batch_indices == batch_idx

        batch_gt = ground_truth[batch_mask].reshape(window, -1, 2).transpose(1, 0, 2)
        batch_pred = predictions[batch_idx].cpu().numpy()
        ade = 0
        fde = 0
        traj_ades = []
        for j in range(len(batch_gt)):
            gt_traj = batch_gt[j]
            pred_traj = batch_pred[j]

            # Split into history and future
            gt_history = gt_traj[: history_frames + 1]
            gt_future = gt_traj[history_frames:]
            pred_future = pred_traj[history_frames:]

            ade_j = np.sqrt(np.sum((pred_future - gt_future) ** 2, axis=1))
            traj_ades.append(ade_j)
            ade_segments = (ade_j[:-1] + ade_j[1:]) / 2
            fde_j = np.sqrt(np.sum((pred_future[-1] - gt_future[-1]) ** 2))

            ade += ade_j
            fde += fde_j
            max_ade = max(max_ade, ade_j.max())
        ades.append(ade / len(batch_gt))
        fdes.append(fde / len(batch_gt))
        all_ades.append(traj_ades)

    # For each selected batch
    for i, batch_idx in enumerate(selected_batches):
        ax = fig.add_subplot(gs[i])

        # Get all trajectories for this batch
        batch_mask = batch_indices == batch_idx

        batch_gt = ground_truth[batch_mask].reshape(window, -1, 2).transpose(1, 0, 2)
        batch_pred = predictions[batch_idx].cpu().numpy()

        # For each trajectory in this batch
        for j in range(len(batch_gt)):
            gt_traj = batch_gt[j]
            pred_traj = batch_pred[j]

            # Split into history and future
            gt_history = gt_traj[: history_frames + 1]
            gt_future = gt_traj[history_frames:]
            pred_future = pred_traj[history_frames:]

            # Plot history trajectory (gray)
            ax.plot(
                gt_history[:, 0],
                gt_history[:, 1],
                "o-",
                color="tab:gray",
                linewidth=2,
                markersize=4,
                alpha=0.6,
                label="History" if j == 0 else "",
                zorder=-1,
            )

            # Plot ground truth future trajectory (blue)
            ax.plot(
                gt_future[:, 0],
                gt_future[:, 1],
                "o-",
                color="tab:blue",
                linewidth=2,
                markersize=6,
                label="Ground Truth" if j == 0 else "",
                alpha=0.9,
                zorder=1,
            )

            # Create colored segments for prediction based on ADE
            points = pred_future.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Normalize ADE values for color mapping
            norm = plt.Normalize(0, max(1.0, max_ade))

            # Create colored line collection
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)  # type: ignore
            lc.set_array(ade_segments)
            line = ax.add_collection(lc)

            # Plot prediction points
            scatter = ax.scatter(
                pred_future[:, 0],
                pred_future[:, 1],
                c=all_ades[i][j],
                cmap=cmap,
                norm=norm,
                s=32,
                # edgecolor="black",
                linewidth=0.5,
                label="Prediction" if j == 0 else "",
                zorder=3,
            )

        ax.set_aspect("equal")

        # Set title and labels
        # ax.set_title(f"(ADE {float(ades[i]):.2f}, FDE {float(fdes[i]):.2f})", fontsize=10)
        ax.set_xlabel("X Position", fontsize=10)
        ax.set_ylabel("Y Position", fontsize=10)
        ax.grid(True, alpha=0.4)

        # Add legend (only for the first trajectory in each batch)
        # if i == 0:
        # ax.legend(loc="upper right")

    # Set equal aspect ratio
    x_min_all, x_max_all = 0, 0
    y_min_all, y_max_all = 0, 0
    for ax in fig.axes:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_min_all = min(x_min_all, x_min)
        x_max_all = max(x_max_all, x_max)
        y_min_all = min(y_min_all, y_min)
        y_max_all = max(y_max_all, y_max)
    x_mid = (x_max_all + x_min_all) / 2
    y_mid = (y_max_all + y_min_all) / 2
    max_range = (max(x_max_all - x_min_all, y_max_all - y_min_all) / 2) * 1.1
    for ax in fig.axes:
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)

    # Add a colorbar for ADE
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # type: ignore
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Displacement Error")

    # Make axis thicker
    for ax in fig.axes:
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)

    plt.tight_layout(rect=[0, 0, 0.90, 1])  # type: ignore
    # plt.subplots_adjust(hspace=0.4)

    # plt.show()

    plt.savefig("figures/trajectories_eth_4.svg")

    return fig


def evaluate_social_forces(original: np.ndarray, scenarios: dict, config: dict) -> dict:
    """Evaluate the social forces by looking at the vector fields of different
    scenario's compared to the original trajectory.

    Args:
        original: Tensor of shape (trajs, frames, 2) containing the original trajectory
        scenarios: Dictionary containing the scenarios to evaluate, lists inside
        each with shape (trajs, frames, 2)
        config: Configuration dictionary

    Returns:
        Dictionary containing the social forces for each scenario
    """

    forces = np.zeros((len(scenarios), original.shape[1], 2))

    for scenario in scenarios:
        scenario_forces = np.zeros((original.shape[0], original.shape[1], 2))
        for field in scenario:
            scenario_forces += field - original
        forces[scenario] = scenario_forces / len(scenario)

    return {scenario: forces[i] for i, scenario in enumerate(scenarios)}


def calculate_collisions(
    trajectories: np.ndarray, config: dict, coll_dist: float = 0.4
) -> list[int]:
    """Calculates collision events for each trajectory with pedestrians sizes and the time between collision events

    Parameters:
    trajectories: np.ndarray
        Array of shape (n_trajs, frames, 2) containing the trajectories
    config: dict
        Configuration dictionary
    coll_dist: float
        Collision distance
    Returns:
        Dictionary containing the collision events and intercollision times
    """

    n_maxtrajectories, n_frames, _ = trajectories.shape

    # Track collision events for each pedestrian
    collision_events = []

    previous_collisions = []
    # Check each frame for collisions
    for frame in range(n_frames):
        # Get positions of all pedestrians in current frame
        positions = trajectories[:, frame, :]

        # Calculate pairwise distances between all pedestrians
        if len(positions) > 1:  # Need at least 2 pedestrians to have collisions
            distances = squareform(pdist(positions, metric="euclidean"))

            # Set diagonal to infinity to avoid self-collisions
            np.fill_diagonal(distances, np.inf)

            # Find all collision pairs (where distance < collision_distance)
            collision_pairs = np.argwhere(distances < coll_dist)

            # Record collision events
            for i, j in collision_pairs:
                if (i, j) not in previous_collisions and (j, i) not in previous_collisions:
                    collision_events.append(frame)
                    previous_collisions.append((i, j))

            # Record previous collisions
            previous_collisions = collision_pairs.tolist()

    return collision_events


def calculate_n_n_relative_position(x: np.ndarray, config: dict) -> np.ndarray:
    """Calculates the relative position of the nearest neighbour
    Args:
        x: Numpy array of shape (trajs, frames, 2)
    """

    if x.shape[0] == 1:
        return np.empty((0, 2))

    vel_filter = 0.25  # m/s

    relative_positions = []
    for frame in range(1, x.shape[1]):
        pairs = squareform(pdist(x[:, frame], metric="euclidean"))
        np.fill_diagonal(pairs, np.inf)
        min_dists = np.argmin(pairs, axis=1)
        rel_coords = x[:, frame] - x[min_dists, frame]
        vels = x[:, frame] - x[:, frame - 1]
        norms = np.linalg.norm(vels, axis=1)
        rel_coords = rel_coords[norms > vel_filter]
        vels = vels[norms > vel_filter]
        # For each pedestrian, align relative position with their heading
        for i in range(len(rel_coords)):
            heading = vels[i] / np.linalg.norm(vels[i])  # (heading direction)
            # Create rotation matrix to align with heading
            theta = np.arctan2(heading[1], heading[0]) - np.pi / 2  # forward is up
            rotation_matrix = np.array(
                [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
            )
            rel_coords[i] = np.dot(rotation_matrix, rel_coords[i])
        if len(rel_coords > 0):
            rel_coords = rel_coords[~np.isnan(rel_coords).any(axis=1)]
            relative_positions.append(rel_coords.reshape(-1, 2))

    return np.vstack(relative_positions) if len(relative_positions) > 0 else np.empty((0, 2))


def calculate_intercollision_time(scenarios: list, config: dict) -> tuple[list, list, int]:
    """Calculates the mean time between subsequent collision event for each trajectory for pedestrian densities

    Parameters:
    scenarios: list
        List of scenarios to evaluate
    config: dict
        Configuration dictionary
    Returns:
        Dictionary containing the collision events and intercollision times

    """
    num_pedestrians = [len(scenario) for scenario in scenarios]
    bins = list(np.linspace(1, max(num_pedestrians), 10))
    scenarios = sorted(scenarios, key=lambda x: len(x))
    num_collisions = [0 for _ in range(len(bins))]
    lengths = [0 for _ in range(len(bins))]

    for scenario in scenarios:
        trajs = scenario[:, config["cond_frames"] :].numpy()
        collision_times = calculate_collisions(trajs, config, coll_dist=0.4)
        bin = np.digitize(len(scenario), bins)
        num_collisions[bin - 1] += len(collision_times)
        lengths[bin - 1] += len(scenario)

    times = [
        1
        / (num_collisions[i] / (lengths[i] * (config["time_window"] - config["cond_frames"])))
        if num_collisions[i] > 0 and lengths[i] > 0
        else 0
        for i in range(len(bins))
    ]

    return bins, times, sum(num_collisions)


def plot_n_n_relative_position_heatmap(
    rel_pos: np.ndarray,
    rel_pos_gt: np.ndarray,
    config: dict,
    rel_pos_prior: np.ndarray | None = None,
):
    if rel_pos_prior is not None:
        fig, axis = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axis = plt.subplots(1, 2, figsize=(13, 6))

    radius = 1.5

    if rel_pos.shape[0] < 100 or rel_pos_gt.shape[0] < 100:
        return fig

    x, y = rel_pos[:, 0], rel_pos[:, 1]
    x_gt, y_gt = rel_pos_gt[:, 0], rel_pos_gt[:, 1]
    if rel_pos_prior is not None:
        x_prior, y_prior = rel_pos_prior[:, 0], rel_pos_prior[:, 1]
    x_grid = np.linspace(-radius, radius, 100)
    y_grid = np.linspace(-radius, radius, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    kde = stats.gaussian_kde(np.vstack([x, y]), bw_method=0.05)
    kde_gt = stats.gaussian_kde(np.vstack([x_gt, y_gt]), bw_method=0.05)
    if rel_pos_prior is not None:
        kde_prior = stats.gaussian_kde(np.vstack([x_prior, y_prior]), bw_method=0.05)
    Z = kde(positions).reshape(X.shape)
    Z_gt = kde_gt(positions).reshape(X.shape)
    if rel_pos_prior is not None:
        Z_prior = kde_prior(positions).reshape(X.shape)
        densities = [Z_prior, Z, Z_gt]
        titles = ["Prior", "Predictions", "Ground Truth"]
    else:
        densities = [Z, Z_gt]
        titles = ["Predictions", "Ground Truth"]

    for i, ax in enumerate(axis):
        # Plot heatmap
        im = ax.imshow(
            densities[i],
            extent=[-radius, radius, -radius, radius],
            origin="lower",
            cmap="plasma",
            aspect="equal",
        )

        plt.colorbar(im, ax=ax, label="Density")

        circle = plt.Circle(
            (0, 0), radius, fill=False, color="black", linewidth=1, linestyle="--"
        )
        ax.add_patch(circle)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        ax.set_xlabel("Relative X Position")
        ax.set_ylabel("Relative Y Position")
        ax.set_title(
            f"Density of Relative Positions of Nearest Neighbours \n {titles[i]} (N={len(rel_pos)}) (Min. 0.25 m/s velocity)"
        )

        # Set limits
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

    plt.tight_layout()
    return fig


def plot_intercollision_time(pred, gt, bins, n_collisions, config):
    fig = plt.figure(figsize=(8, 6))

    # Scatter plot to represent densities with dots
    x_bins = bins
    y_pred = pred
    y_gt = gt

    # Draw small lines between the two histograms points
    plt.vlines(
        x=x_bins,
        ymin=y_pred,
        ymax=y_gt,
        color="black",
        linestyle="--",
        alpha=0.6,
        linewidth=0.5,
    )

    # Plot histograms
    plt.scatter(x_bins, y_pred, color="tab:blue", alpha=0.9, s=55, label="Prediction")
    plt.scatter(x_bins, y_gt, color="tab:red", alpha=0.9, s=55, label="Ground Truth")

    plt.yscale("log")
    plt.xlabel("Number of Pedestrians")
    plt.ylabel("Time between collisions per pedestrian (s/collision)")
    plt.legend()
    plt.title("Intercollision time analysis")

    plt.text(
        0.58,
        0.98,
        f"Pred. #Collisions: {n_collisions[0]} \n GT. #Collisions: {n_collisions[1]}",
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, color="white"),
        alpha=0.8,
    )

    return fig


def plot_nfe_performance(solvers):
    nfes = [[2, 5, 20, 50, 100], [2, 5, 20, 50, 100]]
    ades = [
        [
            0.3111835026219487,
            0.12105424567428417,
            0.11454735894501208,
            0.12227137644938194,
            0.11537451251759194,
        ],
        [
            0.5709405770525336,
            0.12357093611080198,
            0.11306273861625231,
            0.1207973239282146,
            0.11941198342829011,
        ],
    ]

    times = [
        [
            0.051655030250549315 * 2,
            1.3537556409835816,
            7.418587970733642,
            19.518195223808288,
            39.2497216463089,
        ],
        [
            0.1170501470565796 * 2,
            6.466171407699585,
            26.50962793827057,
            62.46338038444519,
            138.4788780450821,
        ],
    ]

    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["tab:blue", "tab:orange"]

    # Normalize time for sizing the markers
    all_times = [t for sublist in times for t in sublist]
    min_time, max_time = min(all_times), max(all_times)
    min_size, max_size = 50, 200  # Marker size range

    # Plot for each solver
    for i, (solver, ade, nfe, time) in enumerate(zip(solvers, ades, nfes, times)):
        color = colors[i]

        # Connect the dots with dotted lines
        ax.plot(nfe, ade, color=color, linewidth=1.5, alpha=0.7, zorder=-1)

        # Plot NFE vs ADE with time represented by marker size
        scatter = ax.scatter(
            nfe,
            ade,
            s=75,
            c=[color],
            # alpha=0.9,
            edgecolors="black",
            linewidths=1,
            label=solver,
        )

    # Add time text to each point
    for i in range(len(times[0])):
        x = nfe[i]
        if i == 0:
            x *= 4
        elif i == 1:
            x *= 1.85
        ax.text(
            x,
            ade[i] * (1.5 if i > 0 else 0.55),
            f"{times[0][i]:.2f}s",
            fontsize=13,
            color=colors[0],
            horizontalalignment="center",
            verticalalignment="center",
            # path_effects=[
            # pe.withStroke(linewidth=1, foreground="black"),
            # ],
        )
        ax.text(
            x,
            ade[i] * (1.25 if i > 0 else 1),
            f"{times[1][i]:.2f}s",
            fontsize=13,
            color=colors[1],
            horizontalalignment="center",
            verticalalignment="center",
            # path_effects=[
            # pe.withStroke(linewidth=1, foreground="black"),
            # ],
        )

    # Set labels and title
    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=15)
    ax.set_ylabel("Average Displacement Error (ADE)", fontsize=15)

    # Made axis thicker
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)

    # Increas tick size
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_xticks([2, 5, 20, 50, 100])
    ax.set_xlim([-2, 110])

    ax.grid(axis="y", color="gray", linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(loc="upper right", fontsize=12)

    # Tight layout
    plt.tight_layout()
    plt.savefig("figures/nfe_performance.svg")
    plt.show()

    return fig


def old_evaluate(ground_truths, predictions, config, debugging: bool) -> None:
    max_nodes = max([len(graph.matched_pos) for sample in ground_truths for graph in sample])

    ground_truth_pos = np.array(
        [
            [
                np.pad(
                    graph.matched_pos.numpy(),
                    ((0, max_nodes - len(graph.matched_pos)), (0, 0)),
                    constant_values=np.nan,
                )
                for graph in sample
            ]
            for sample in ground_truths
        ]
    )

    predicted_pos = np.array(
        [
            [
                np.pad(
                    graph.matched_pos.numpy(),
                    ((0, max_nodes - len(graph.matched_pos)), (0, 0)),
                    constant_values=np.nan,
                )
                for graph in sample
            ]
            for sample in predictions
        ]
    )

    trajectory_anim = animate_cell_trajectories(
        ground_truth_pos, predicted_pos, predictions, config, debugging
    )

    centered_trajectory_plot = plot_centered_trajectories(
        ground_truth_pos, predicted_pos, 100, len(ground_truths), 20, debugging
    )

    # Dont evaluate on first step
    ground_truth_pos = ground_truth_pos[:, 1:, :]
    predicted_pos = predicted_pos[:, 1:, :]

    FDEs = final_displacement_error(ground_truth_pos, predicted_pos)
    relative_FDEs = relative_final_displacement_error(ground_truth_pos, predicted_pos)
    cum_FDEs = cumulative_displacement_error(ground_truth_pos, predicted_pos)
    one_step_DEs = one_step_displacement_error(ground_truth_pos, predicted_pos)

    # avg_error_plot = plot_average_displacement_error(ground_truth, predictions)
    # final_error_plot = plot_final_displacement_error(ground_truth_pos, predicted_pos)
    # error_cdf_plot = plot_error_cdf(ground_truth, predictions)

    # directional_error_plot = plot_directional_error(ground_truth, predictions)

    if not debugging:
        wandb.log(
            {  # Scalar metrics
                # "average_displacement_errors": np.mean(ADEs),
                "final_displacement_errors": np.mean(FDEs),
                "relative_final_displacement_errors": np.mean(relative_FDEs),
                "cumulative_displacement_errors": np.mean(cum_FDEs),
                "one_step_displacement_errors": np.mean(one_step_DEs),
            }
        )

        wandb.log(
            {  # Plots and animations
                # "cell_trajectories": wandb.Image(
                #     trajectory_plot, caption="Cell Trajectory Rollouts"
                # ),
                "cell_trajectories_animation": wandb.Video(
                    trajectory_anim,  # type: ignore
                    fps=3,
                    format="gif",
                ),  # type: ignore
                "cell_trajectories_centered": wandb.Image(
                    centered_trajectory_plot, caption="Centered Trajectory Rollouts"
                ),
                # "average_displacement_error": wandb.Image(
                #     avg_error_plot, caption="Average Displacement Error"
                # ),
                # "final_displacement_error": wandb.Image(
                #     final_error_plot, caption="Final Displacement Error"
                # ),
                # "error_cdf": wandb.Image(
                #     error_cdf_plot, caption="Cumulative Distribution of Displacement Errors"
                # ),
                # "directional_error": wandb.Image(
                #     directional_error_plot, caption="Distribution of Angular Errors"
                # ),
                # "error_heatmap": wandb.Image(
                #     error_heatmap_plot, caption="Spatial Distribution of Prediction Errors"
                # ),
            }
        )

        for file in [
            # trajectory_plot,
            trajectory_anim,
            centered_trajectory_plot,
            # avg_error_plot,
            # final_error_plot,
            # error_cdf_plot,
            # directional_error_plot,
            # error_heatmap_plot,
        ]:
            os.unlink(file)


def evaluate_and_plot(x0, x, x1, config, debugging: bool) -> tuple[dict, dict]:
    figs, metrics = {}, {}

    # flows = plot_flows(x0, x1, x, model, config, debugging=debugging)
    # figs["flows"] = flows
    plt.style.use("seaborn-v0_8-paper")

    traj_evolution = plot_trajectory_evolution(
        x0,
        x,
        x1,
        fig_size=(18, 6),
        max_trajectories=10,
        # cond_length=config["cond_frames"],
        max_x_and_y=config["max_x_and_y"],
    )
    figs["trajectory_evolution"] = traj_evolution

    x0_vel, x_vel, x1_vel = calculate_velocity_magnitude(x0, x, x1)
    # figs["velocity"] = make_violin_box_plot(x0_vel, x_vel, x1_vel, "Velocity")
    figs["velocity_kde"] = make_kde_plot(x0_vel, x_vel, x1_vel, "Velocity", min=0, max=1)
    metrics["velocity_wasserstein"] = stats.wasserstein_distance(x_vel, x1_vel)

    x0_acc, x_acc, x1_acc = calculate_acceleration(x0, x, x1)
    # figs["acceleration"] = make_violin_box_plot(x0_acc, x_acc, x1_acc, "Acceleration")
    figs["acceleration_kde"] = make_kde_plot(
        x0_acc, x_acc, x1_acc, "Acceleration", min=0, max=0.5
    )
    metrics["acceleration_wasserstein"] = stats.wasserstein_distance(x_acc, x1_acc)

    x0_tort = calculate_tortuosity(x0.cpu().numpy())
    x_tort = calculate_tortuosity(x.cpu().numpy())
    x1_tort = calculate_tortuosity(x1.cpu().numpy())
    # figs["tortuosity"] = make_violin_box_plot(x0_tort, x_tort, x1_tort, "Tortuosity")
    figs["tortuosity_kde"] = make_kde_plot(
        x0_tort, x_tort, x1_tort, "Tortuosity", min=0, max=10
    )
    metrics["tortuosity_wasserstein"] = stats.wasserstein_distance(x_tort, x1_tort)

    x0_ang_vel, x_ang_vel, x1_ang_vel = calculate_angular_velocity(x0, x, x1)
    # figs["angular_velocity"] = make_violin_box_plot(
    #     x0_ang_vel, x_ang_vel, x1_ang_vel, "Angular Velocity"
    # )
    figs["angular_velocity_kde"] = make_kde_plot(
        x0_ang_vel, x_ang_vel, x1_ang_vel, "Angular Velocity"
    )
    metrics["angular_velocity_wasserstein"] = stats.wasserstein_distance(x_ang_vel, x1_ang_vel)

    x0_msds = mean_square_displacement(x0.cpu().numpy(), list(range(0, x.shape[1])))
    x_msds = mean_square_displacement(x.cpu().numpy(), list(range(0, x.shape[1])))
    x1_msds = mean_square_displacement(x1.cpu().numpy(), list(range(0, x.shape[1])))
    msd_plot = plot_mean_square_displacement(x0_msds, x_msds, x1_msds, 10)
    figs["mean_square_displacement"] = msd_plot
    metrics["mean_square_displacement_wasserstein"] = stats.wasserstein_distance(
        x_msds, x1_msds
    )

    x0_autocorr, x0_std = velocity_autocorrelation(
        x0.cpu().numpy(), list(range(0, x0.shape[1] - 1))
    )
    x_autocorr, x_std = velocity_autocorrelation(
        x.cpu().numpy(), list(range(0, x.shape[1] - 1))
    )
    x1_autocorr, x1_std = velocity_autocorrelation(
        x1.cpu().numpy(), list(range(0, x1.shape[1] - 1))
    )
    figs["velocity_autocorrelation"] = plot_velocity_autocorrelation(
        x0_autocorr, x_autocorr, x1_autocorr, 10
    )
    metrics["velocity_autocorrelation_wasserstein"] = stats.wasserstein_distance(
        x_autocorr, x1_autocorr
    )

    # distances_over_time_x0, std = calculate_neighbour_dist_over_time(x0.cpu(), config["knn"])
    # distances_over_time, std = calculate_neighbour_dist_over_time(x.cpu(), config["knn"])
    # distances_over_time_gt, std_gt = calculate_neighbour_dist_over_time(
    #     x1.cpu(), config["knn"]
    # )
    # figs["neighbour_dist"] = plot_neighbour_dist_over_time(
    #     distances_over_time, std, distances_over_time_gt, std_gt, distances_over_time_x0
    # )
    # metrics["neighbour_dist_wasserstein"] = stats.wasserstein_distance(
    #     distances_over_time, distances_over_time_gt
    # )

    # neighbourhood_vel_corr = calculate_neighbourhood_velocity_correlation(
    #     x.cpu(), config["knn"]
    # )
    # neighbourhood_vel_corr_gt = calculate_neighbourhood_velocity_correlation(
    #     x1.cpu(), config["knn"]
    # )
    # neighbourhood_vel_corr_x0 = calculate_neighbourhood_velocity_correlation(
    #     x0.cpu(), config["knn"]
    # )
    # figs["local_velocity_correlation"] = plot_local_velocity_correlation(
    #     neighbourhood_vel_corr, neighbourhood_vel_corr_gt, neighbourhood_vel_corr_x0
    # )
    # local_vel_corr_aggr = np.mean(neighbourhood_vel_corr, axis=1)
    # local_vel_corr_gt_aggr = np.mean(neighbourhood_vel_corr_gt, axis=1)
    # metrics["neightbourhood_velocity_correlation_wasserstein"] = stats.wasserstein_distance(
    #     local_vel_corr_aggr, local_vel_corr_gt_aggr
    # )

    # if debugging:
    #     plt.show()

    return figs, metrics


def only_evaluate(x0, x, x1, config) -> dict:
    metrics = {}

    metrics["position"] = [x0.cpu().numpy(), x.cpu().numpy(), x1.cpu().numpy()]

    x0_vel = np.diff(x0.cpu().numpy(), axis=1)
    x_vel = np.diff(x.cpu().numpy(), axis=1)
    x1_vel = np.diff(x1.cpu().numpy(), axis=1)
    metrics["velocity"] = [x0_vel, x_vel, x1_vel]

    x0_vel, vel, gt_vel = calculate_velocity_magnitude(x0, x, x1)
    metrics["velocity_magnitude"] = [x0_vel, vel, gt_vel]
    metrics["velocity_wasserstein"] = stats.wasserstein_distance(
        vel.flatten(), gt_vel.flatten()
    )

    x0_acc, acc, gt_acc = calculate_acceleration(x0, x, x1)
    metrics["acceleration"] = [x0_acc, acc, gt_acc]
    metrics["acceleration_wasserstein"] = stats.wasserstein_distance(
        acc.flatten(), gt_acc.flatten()
    )

    x0_tort = calculate_tortuosity(x0.cpu().numpy())
    pred_tort = calculate_tortuosity(x.cpu().numpy())
    gt_tort = calculate_tortuosity(x1.cpu().numpy())
    metrics["tortuosity"] = [x0_tort, pred_tort, gt_tort]
    metrics["tortuosity_wasserstein"] = stats.wasserstein_distance(pred_tort, gt_tort)

    x0_ang_vel, pred_ang_vel, gt_ang_vel = calculate_angular_velocity(x0, x, x1)
    metrics["angular_velocity"] = [x0_ang_vel, pred_ang_vel, gt_ang_vel]
    metrics["angular_velocity_wasserstein"] = stats.wasserstein_distance(
        pred_ang_vel.flatten(), gt_ang_vel.flatten()
    )

    # x0_vel_autocorr, _ = velocity_autocorrelation(
    #     x0.cpu().numpy(), list(range(0, x0.shape[1] - 1))
    # )
    # vel_autocorr, _ = velocity_autocorrelation(x.cpu().numpy(), list(range(0, x.shape[1] - 1)))
    # gt_vel_autocorr, _ = velocity_autocorrelation(
    #     x1.cpu().numpy(), list(range(0, x1.shape[1] - 1))
    # )
    # metrics["velocity_autocorrelation"] = [x0_vel_autocorr, vel_autocorr, gt_vel_autocorr]
    # metrics["velocity_autocorrelation_wasserstein"] = stats.wasserstein_distance(
    #     vel_autocorr, gt_vel_autocorr
    # )

    # x0_msd = mean_square_displacement(x0.cpu().numpy(), list(range(0, x0.shape[1])))
    # x_msd = mean_square_displacement(x.cpu().numpy(), list(range(0, x.shape[1])))
    # gt_msd = mean_square_displacement(x1.cpu().numpy(), list(range(0, x1.shape[1])))
    # metrics["mean_square_displacement"] = [x0_msd, x_msd, gt_msd]
    # metrics["mean_square_displacement_wasserstein"] = stats.wasserstein_distance(x_msd, gt_msd)

    # distances_over_time_x0, x0_std = calculate_neighbour_dist_over_time(
    #     x0.cpu(), config["knn"]
    # )
    # distances_over_time, std = calculate_neighbour_dist_over_time(x.cpu(), config["knn"])
    # distances_over_time_gt, std_gt = calculate_neighbour_dist_over_time(
    #     x1.cpu(), config["knn"]
    # )

    # metrics["neighbour_dist"] = [
    #     distances_over_time_x0,
    #     distances_over_time,
    #     distances_over_time_gt,
    # ]
    # if len(distances_over_time) > 0:
    #     metrics["neighbour_dist_wasserstein"] = stats.wasserstein_distance(
    #         distances_over_time, distances_over_time_gt
    #     )

    # neighbourhood_vel_corr_x0 = calculate_neighbourhood_velocity_correlation(
    #     x0.cpu(), config["knn"]
    # )
    # neighbourhood_vel_corr = calculate_neighbourhood_velocity_correlation(
    #     x.cpu(), config["knn"]
    # )
    # neighbourhood_vel_corr_gt = calculate_neighbourhood_velocity_correlation(
    #     x1.cpu(), config["knn"]
    # )
    # if len(neighbourhood_vel_corr) > 0:
    #     local_vel_corr_aggr_x0 = np.nanmean(neighbourhood_vel_corr_x0, axis=1)
    #     local_vel_corr_aggr = np.nanmean(neighbourhood_vel_corr, axis=1)
    #     local_vel_corr_gt_aggr = np.nanmean(neighbourhood_vel_corr_gt, axis=1)
    #     # metrics["local_velocity_correlation"] = [
    #     #     local_vel_corr_aggr_x0,
    #     #     local_vel_corr_aggr,
    #     #     local_vel_corr_gt_aggr,
    #     # ]
    #     metrics["neightbourhood_velocity_correlation_wasserstein"] = (
    #         stats.wasserstein_distance(local_vel_corr_aggr, local_vel_corr_gt_aggr)
    #     )

    metrics["nearest_neighbour_distance"] = [
        calculate_nearest_neighbour_dist(x0[:, config["cond_frames"] :].cpu().numpy(), config),
        calculate_nearest_neighbour_dist(x[:, config["cond_frames"] :].cpu().numpy(), config),
        calculate_nearest_neighbour_dist(x1[:, config["cond_frames"] :].cpu().numpy(), config),
    ]

    if config["dims"] == 2:
        metrics["nearest_neighbour_relative_position"] = [
            calculate_n_n_relative_position(
                x0[:, config["cond_frames"] :].cpu().numpy(), config
            ),
            calculate_n_n_relative_position(
                x[:, config["cond_frames"] :].cpu().numpy(), config
            ),
            calculate_n_n_relative_position(
                x1[:, config["cond_frames"] :].cpu().numpy(), config
            ),
        ]
    else:
        metrics["nearest_neighbour_relative_position"] = [[], [], []]

    metrics["intercollision_time"] = []
    metrics["pedestrian_density"] = []

    return metrics


def plot_evaluation(data: dict, config: dict, testing: bool) -> dict:
    figs = {}
    bs = config["test_size"] if testing else 1
    c_size = config["cond_frames"]

    # figs["x_position_hist"] = make_hist_plot(
    #     data["position"][1][:, c_size:, 0],
    #     data["position"][2][:, c_size:, 0],
    #     title="X Position",
    # )
    # figs["y_position_hist"] = make_hist_plot(
    #     data["position"][1][:, c_size:, 1],
    #     data["position"][2][:, c_size:, 1],
    #     title="Y Position",
    # )

    # figs["x_velocity_hist"] = make_hist_plot(
    #     data["velocity"][1][:, c_size - 1 :, 0],
    #     data["velocity"][2][:, c_size - 1 :, 0],
    #     "X Velocity",
    #     prior=data["velocity"][0][:, c_size - 1 :, 0],
    #     min_max=(-3, 3) if not "SDD" in config["processed_path"] else None,
    # )
    # figs["x_velocity_hist_log"] = make_log_hist_plot(
    #     data["velocity"][1][:, c_size - 1 :, 0],
    #     data["velocity"][2][:, c_size - 1 :, 0],
    #     "X Velocity",
    #     min_max=(-3, 3) if not "SDD" in config["processed_path"] else None,
    # )

    # figs["y_velocity_hist"] = make_hist_plot(
    #     data["velocity"][1][:, c_size - 1 :, 1],
    #     data["velocity"][2][:, c_size - 1 :, 1],
    #     "Y Velocity",
    #     prior=data["velocity"][0][:, c_size - 1 :, 1],
    #     min_max=(-3, 3) if not "SDD" in config["processed_path"] else None,
    # )
    # figs["y_velocity_hist_log"] = make_log_hist_plot(
    #     data["velocity"][1][:, c_size - 1 :, 1],
    #     data["velocity"][2][:, c_size - 1 :, 1],
    #     "Y Velocity",
    #     min_max=(-3, 3) if not "SDD" in config["processed_path"] else None,
    # )

    # figs["velocity_magnitude_hist"] = make_hist_plot(
    #     np.linalg.norm(data["velocity"][1][:, c_size - 1 :], axis=2),
    #     np.linalg.norm(data["velocity"][2][:, c_size - 1 :], axis=2),
    #     "Velocity Magnitude",
    #     min_max=(0, 4) if not "SDD" in config["processed_path"] else None,
    #     prior=np.linalg.norm(data["velocity"][0][:, c_size - 1 :], axis=2),
    # )
    # figs["velocity_magnitude_hist_log"] = make_log_hist_plot(
    #     np.linalg.norm(data["velocity"][1][:, c_size - 1 :], axis=2),
    #     np.linalg.norm(data["velocity"][2][:, c_size - 1 :], axis=2),
    #     "Velocity Magnitude",
    #     min_max=(0, 4) if not "SDD" in config["processed_path"] else None,
    # )

    # figs["acceleration_hist"] = make_hist_plot(
    #     data["acceleration"][1][:, c_size - 2 :],
    #     data["acceleration"][2][:, c_size - 2 :],
    #     "Acceleration Magnitude",
    #     prior=data["acceleration"][0][:, c_size - 2 :],
    #     min_max=(0, 3) if not "SDD" in config["processed_path"] else None,
    # )
    # figs["acceleration_hist_log"] = make_log_hist_plot(
    #     data["acceleration"][1][:, c_size - 2 :],
    #     data["acceleration"][2][:, c_size - 2 :],
    #     "Acceleration Magnitude",
    #     min_max=(0, 3) if not "SDD" in config["processed_path"] else None,
    # )

    # # figs["tortuosity_hist"] = make_hist_plot(
    # #     data["tortuosity"][1],
    # #     data["tortuosity"][2],
    # #     "Tortuosity (5m minimum)",
    # #     min_max=(0, 2) if not "SDD" in config["processed_path"] else None,
    # # )
    # # figs["tortuosity_hist_log"] = make_log_hist_plot(
    # #     data["tortuosity"][1],
    # #     data["tortuosity"][2],
    # #     "Tortuosity (5m minimum)",
    # #     min_max=(0, 2) if not "SDD" in config["processed_path"] else None,
    # # )

    # figs["angular_velocity_hist"] = make_hist_plot(
    #     data["angular_velocity"][1][:, c_size:],
    #     data["angular_velocity"][2][:, c_size:],
    #     "Angular Velocity Magnitude",
    #     # min=ang_vel_mean - 3 * ang_vel_std,
    #     # max=ang_vel_mean + 3 * ang_vel_std,
    # )

    # # # Plot mean square displacement
    # # msd_x0, msd, msd_gt = data["mean_square_displacement"]
    # # msd_x0 = msd_x0.reshape(bs, -1).mean(axis=0)
    # # msd = msd.reshape(bs, -1).mean(axis=0)
    # # msd_gt = msd_gt.reshape(bs, -1).mean(axis=0)
    # # figs["mean_square_displacement"] = plot_mean_square_displacement(
    # #     msd_x0,
    # #     msd,
    # #     msd_gt,
    # #     1,
    # # )

    # # # Plot velocity autocorrelation
    # # autocorr_x0, autocorr, autocorr_gt = data["velocity_autocorrelation"]
    # # autocorr_x0 = autocorr_x0.reshape(bs, -1).mean(axis=0)
    # # autocorr = autocorr.reshape(bs, -1).mean(axis=0)
    # # autocorr_gt = autocorr_gt.reshape(bs, -1).mean(axis=0)
    # # figs["velocity_autocorrelation"] = plot_velocity_autocorrelation(
    # #     autocorr_x0,
    # #     autocorr,
    # #     autocorr_gt,
    # #     1,
    # # )

    # if (
    #     len(data["nearest_neighbour_distance"][1]) > 0
    #     and len(data["nearest_neighbour_distance"][2]) > 0
    # ):
    #     figs["nearest_neighbour_distance_hist"] = make_hist_plot(
    #         data["nearest_neighbour_distance"][1],
    #         data["nearest_neighbour_distance"][2],
    #         "Nearest Neighbour Distance (m)",
    #         prior=data["nearest_neighbour_distance"][0],
    #     )

    #     figs["nearest_neighbour_distance_hist_log"] = make_log_hist_plot(
    #         data["nearest_neighbour_distance"][1],
    #         data["nearest_neighbour_distance"][2],
    #         "Nearest Neighbour Distance (m)",
    #         prior=data["nearest_neighbour_distance"][0],
    #     )

    # if len(data["nearest_neighbour_relative_position"][1]) > 0:
    #     figs["nearest_neighbour_relative_position"] = plot_n_n_relative_position_heatmap(
    #         data["nearest_neighbour_relative_position"][1],
    #         data["nearest_neighbour_relative_position"][2],
    #         config,
    #         data["nearest_neighbour_relative_position"][0],
    #     )

    # figs["intercollision_time"] = plot_intercollision_time(
    #     data["intercollision_time"][0],
    #     data["intercollision_time"][1],
    #     data["pedestrian_density"],
    #     data["num_collisions"],
    #     config,
    # )

    figs["probabilistic_evaluation"] = make_hist_plot_all_paper(
        [
            data["velocity"][1][:, c_size - 1 :, 0],
            np.linalg.norm(data["velocity"][1][:, c_size - 1 :], axis=2),
            np.linalg.norm(data["velocity"][1][:, c_size - 1 :], axis=2),
            data["velocity"][1][:, c_size - 1 :, 1],
            data["acceleration"][1][:, c_size - 2 :],
            data["acceleration"][1][:, c_size - 2 :],
        ],
        [
            data["velocity"][2][:, c_size - 1 :, 0],
            np.linalg.norm(data["velocity"][2][:, c_size - 1 :], axis=2),
            np.linalg.norm(data["velocity"][2][:, c_size - 1 :], axis=2),
            data["velocity"][2][:, c_size - 1 :, 1],
            data["acceleration"][2][:, c_size - 2 :],
            data["acceleration"][2][:, c_size - 2 :],
        ],
        [
            "X Velocity",
            "Velocity Magnitude",
            "Velocity Magnitude",
            "Y Velocity",
            "Acceleration Magnitude",
            "Acceleration Magnitude",
        ],
    )
    # plt.savefig("figures/probabilistic_evaluation_gravity.svg")

    plt.show()

    return figs


def compute_similarities(x: Tensor, x1: Tensor, config: dict) -> dict:
    """Compute similarities measures between the predicted and ground truth trajectories.
    Args:
        x: Tensor of shape (trajs, frames, 2) containing predicted trajectories
        x1: Tensor of shape (trajs, frames, 2) containing ground truth trajectories
        config: Configuration dictionary"""

    similarities = {
        "ade": 0,
        "fde": 0,
        "nmse": 0,
        # "dynamic_time_warping": 0,
        "frechet": 0,
        "wasserstein": 0,
    }
    x = x[:, config["cond_frames"] :]
    x1 = x1[:, config["cond_frames"] :]

    # Compute average displacement error
    error = torch.norm(x - x1, dim=-1)
    similarities["ade"] = float(torch.mean(error))  # type: ignore

    # Compute final displacement error
    error = torch.norm(x[:, -1] - x1[:, -1], dim=-1)
    similarities["fde"] = float(torch.mean(error))  # type: ignore

    # Compute the Normalized Mean Squared Error (NMSE)
    mse = torch.mean((x - x1) ** 2)
    mean_velocity = torch.mean(torch.norm(torch.diff(x1, dim=1), dim=-1))
    similarities["nmse"] = float(mse / mean_velocity)  # type: ignore

    # Calculate Dynamic Time Warping  and Frechet distance
    dtw_dist, frechet = 0, 0
    for pred, gt in zip(x, x1):
        # dtw_dist += dtw(pred.cpu(), gt.cpu(), distance_only=True).distance  # type: ignore
        frechet += frechet_dist(pred.cpu(), gt.cpu())  # type: ignore
    # similarities["dynamic_time_warping"] = float(dtw_dist / x.shape[0])  # type: ignore
    similarities["frechet"] = float(frechet / x.shape[0])  # type: ignore

    # Calculate Wasserstein distance
    # similarities["wasserstein"] = float(wasserstein_distance(x, x1))  # type: ignore

    return similarities


def compute_ade(x: list, batch: Data, config: dict) -> float:
    """Compute Average Displacement Error between the predicted and ground truth trajectories.
    Args:
        x: List of Tensors of shape (trajs, frames, dims) containing predicted trajectories
        x1: List of Tensors of shape (trajs, frames, dims) containing ground truth trajectories
        config: Configuration dictionary"""

    ades = []
    for i, pred in enumerate(x):
        gt_batch = batch.get_example(i)
        gt = gt_batch.trajectories.view(config["time_window"], -1, config["dims"]).transpose(
            0, 1
        )
        predictions = pred[:, config["cond_frames"] :]
        ground_truth = gt[:, config["cond_frames"] :].cpu()
        ades.append(float(torch.mean(torch.norm(predictions - ground_truth, dim=-1))))

    return sum(ades) / len(ades)
