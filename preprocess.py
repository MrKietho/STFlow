import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import torch
from torch import Tensor
import os
from tqdm import tqdm


def read_recordings(path: str, features: list[str]) -> pd.DataFrame:
    all_recordings_df = pd.read_csv(path, header=[0])

    # Map File_name to recording number
    recording_name_to_nr = {
        recording_name: recording_nr
        for recording_nr, recording_name in enumerate(all_recordings_df["File_name"].unique())
    }

    all_recordings_df["RECORDING"] = all_recordings_df["File_name"].map(recording_name_to_nr)

    return all_recordings_df[["RECORDING", "TRACK_ID", "FRAME"] + features]


def read_recordings_boids(path: str) -> np.ndarray:
    all_boids = np.zeros((26, 1000, 25, 2), dtype=np.float32)
    paths = [f"{path}_{i}.npy" for i in range(26)]

    for i, path in enumerate(paths):
        boids = np.load(path)[:, :, :2].astype(np.float32)
        all_boids[i] = boids

    all_boids = all_boids.transpose(0, 2, 1, 3)  # (samples, n_boids, n_frames, 2)

    return all_boids


def sliding_window(
    data: np.ndarray, config: dict, window_size: int, step: int
) -> list[np.ndarray]:
    """Applies sliding window to data of shape (n_trajectories, n_frames, 2)
    to create subsequences of given window size"""

    num_frames = data.shape[1]
    new_num_frames = num_frames - window_size + 1
    samples = []

    for i in tqdm(range(0, new_num_frames, step), desc="Sliding over data using window"):
        window = data[:, i : i + window_size]
        # Drop trajectories with too many NaNs
        nan_mask = (
            np.isnan(window).sum(axis=(1, 2))
            <= config["preprocess_nans_in_trajectory_threshold"] * window_size
        )
        if nan_mask.sum() == 0:
            continue

        samples.append(window[nan_mask])

    return samples


def read_and_process_pedestrians_ETH(config: dict) -> list[np.ndarray]:
    paths = config["data_paths"]
    paths = [paths] if isinstance(paths, str) else paths
    samples = []

    scene_map = {
        "biwi_eth": 0,  # ETH split
        "hotel": 1,  # Hotel split
        "zara01": 2,  # Zara 01 split
        "zara02": 3,  # Zara 02 split
        "zara03": -1,  # Zara 03 split (not used in testing)
        "students001": 4,  # Univ split
        "students003": 4,  # Univ split
        "uni_examples": 4,  # Univ split
    }

    for path in paths:
        # If path is a directory, read all files in directory
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path)]  # type: ignore
        else:
            files = [path]

        for scene, file in enumerate(files):
            seperator = "\t"
            df = pd.read_csv(file, sep=seperator, index_col=False, header=None)
            df.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
            df["frame_id"] = pd.to_numeric(df["frame_id"], downcast="integer")
            df["track_id"] = pd.to_numeric(df["track_id"], downcast="integer")

            df["pos_x"] = df["pos_x"] - df["pos_x"].mean()
            df["pos_y"] = df["pos_y"] - df["pos_y"].mean()

            df.sort_values(["frame_id"], inplace=True)
            track_ids = df["track_id"].unique()
            frame_ids = df["frame_id"].unique()
            for name in scene_map.keys():
                if name in file:
                    df["scene"] = scene_map[name]
                    break
            num_trajs = len(track_ids)
            num_frames = len(frame_ids)
            trajectories = np.full((num_trajs, num_frames, 3), np.nan, dtype=np.float32)

            track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}
            frame_id_to_index = {fid: i for i, fid in enumerate(frame_ids)}

            for _, row in df.iterrows():
                t_idx = track_id_to_index[row["track_id"]]
                f_idx = frame_id_to_index[row["frame_id"]]
                trajectories[t_idx, f_idx] = [row["pos_x"], row["pos_y"], row["scene"]]

            samples.extend(
                sliding_window(
                    trajectories,
                    config,
                    config["time_window"],
                    config["preprocess_sliding_window_step"],
                )
            )

    return samples


def read_and_process_SDD(config: dict) -> list[np.ndarray]:
    paths = config["data_paths"]
    paths = [paths] if isinstance(paths, str) else paths
    samples = []
    scenes = {
        "bookstore": 0,
        "coupa": 1,
        "deathCircle": 2,
        "gates": 3,
        "hyang": 4,
        "little": 5,
        "nexus": 6,
        "quad": 7,
    }

    for path in paths:
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if "annotations.txt" in file:
                    all_files.append(os.path.join(root, file))

        for file in all_files:
            scene = [key for key in scenes.keys() if key in file][0]
            seperator = " "
            class_dic = {
                "Biker": 0,
                "Pedestrian": 1,
                "Skater": 2,
                "Cart": 3,
                "Car": 4,
                "Bus": 5,
            }
            df = pd.read_csv(file, sep=seperator, index_col=False, header=None)
            df.columns = [
                "track_id",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "frame",
                "lost",
                "occluded",
                "generated",
                "label",
            ]
            # df["x"] = df["x_min"] + (df["x_max"] - df["x_min"]) / 2
            # df["y"] = df["y_min"] + (df["y_max"] - df["y_min"]) / 2
            df["x"] = (df["x_max"] + df["x_min"]) / 2
            df["y"] = (df["y_max"] + df["y_min"]) / 2
            df = df[df["lost"] == 0]
            df = df[(df["frame"] % 12) == 0]
            df["frame"] -= df["frame"].min()

            df["x"] = df["x"] / 100
            df["y"] = df["y"] / 100

            if config["only_pedestrians"]:
                df = df[df["label"] == "Pedestrian"]
            df["class"] = df["label"].map(class_dic)

            df.sort_values(["track_id", "frame"], inplace=True)
            track_ids = df["track_id"].unique()
            frame_ids = df["frame"].unique()
            num_trajs = len(track_ids)
            num_frames = len(frame_ids)
            trajectories = np.full((num_trajs, num_frames, 4), np.nan, dtype=np.float32)

            track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}
            frame_id_to_index = {fid: i for i, fid in enumerate(frame_ids)}

            for _, row in df.iterrows():
                t_idx = track_id_to_index[row["track_id"]]
                f_idx = frame_id_to_index[row["frame"]]
                trajectories[t_idx, f_idx] = [row["x"], row["y"], row["class"], scenes[scene]]

            samples.extend(
                sliding_window(
                    trajectories,
                    config,
                    config["time_window"],
                    config["preprocess_sliding_window_step"],
                )
            )

    return samples


def read_and_process_pedestrians_eindhoven(config: dict) -> list[np.ndarray]:
    paths = config["data_paths"]
    paths = [paths] if isinstance(paths, str) else paths
    samples = []

    for path in paths:
        # If path is a directory, read all files in directory
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            files = [path]

        for file in files:
            # Read parquet file
            df = pd.read_parquet(file)
            df.columns = ["frame_id", "track_id", "pos_x", "pos_y"]

            # ms -> s   10 Hz to xHz
            df["frame_id"] = df["frame_id"] // (1000 / config["preprocess_frequency"])
            df = df.drop_duplicates(subset=["frame_id", "track_id"], keep="first")
            df["pos_x"] /= 1000  # mm -> m
            df["pos_y"] /= 1000  # mm -> m
            df["frame_id"] -= df["frame_id"].min()
            # df["pos_x"] -= df["pos_x"].mean()
            # df["pos_y"] -= df["pos_y"].mean()
            df.sort_values(["frame_id"], inplace=True)
            print(f"\nData has {df['track_id'].nunique()} unique track_ids and {len(df)} rows")

            for chunk in tqdm(range(0, 10_000_000, 1_000_000)):
                df_chunk = df.iloc[chunk : chunk + 1_000_000]
                track_ids = df_chunk["track_id"].unique()
                frame_ids = df_chunk["frame_id"].unique()
                num_trajs = len(track_ids)
                num_frames = len(frame_ids)

                trajectories = np.full((num_trajs, num_frames, 2), np.nan, dtype=np.float32)

                track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}
                frame_id_to_index = {fid: i for i, fid in enumerate(frame_ids)}

                for _, row in df_chunk.iterrows():
                    t_idx = track_id_to_index[row["track_id"]]
                    f_idx = frame_id_to_index[row["frame_id"]]
                    trajectories[t_idx, f_idx] = [row["pos_x"], row["pos_y"]]

                samples.extend(
                    sliding_window(
                        trajectories,
                        config,
                        config["time_window"],
                        config["preprocess_sliding_window_step"],
                    )
                )

    return samples


def handle_boundaries(data: np.ndarray) -> np.ndarray:
    """Handle boundaries by splitting trajectories when they reach edge
    Args:
        data: np.ndarray of shape (samples, n_boids, n_frames, n_features)"""

    for sample in range(data.shape[0]):
        for boid in range(data.shape[1]):
            x, y = data[sample, boid, :, 0], data[sample, boid, :, 1]

            # Check if boid reaches edge
            x_diff = np.diff(x)
            y_diff = np.diff(y)

            x_diff = np.insert(x_diff, 0, 0)
            y_diff = np.insert(y_diff, 0, 0)

            x_diff[x_diff > 10] = 0
            x_diff[x_diff < -10] = 0

            y_diff[y_diff > 10] = 0
            y_diff[y_diff < -10] = 0

            x_diff[0] = x[0]
            y_diff[0] = y[0]

            x = np.cumsum(x_diff)
            y = np.cumsum(y_diff)

            data[sample, boid, :, 0] = x
            data[sample, boid, :, 1] = y

    return data


def center_and_scale_boids(data: np.ndarray, max: float = 1.0) -> np.ndarray:
    """Center and scale boids data to [-max, max]
    Args:
        data: np.ndarray of shape (samples, n_boids, n_frames, n_features)
        max: float, max value to scale to"""

    data = data - data.mean(axis=2, keepdims=True)
    data = data / np.max(np.abs(data)) * max

    return data


def construct_boids_graphs(data: np.ndarray, config: dict) -> list[Data]:
    graphs = []

    for sample in range(data.shape[0]):
        trajectories = torch.tensor(data[sample], dtype=torch.float32)
        velocities = torch.diff(
            trajectories, dim=1, append=torch.zeros(trajectories.shape[0], 1, 2)
        )
        sample_graphs = []

        for frame in range(data.shape[2]):
            edge_indices = knn_graph(trajectories[:, frame], k=config["knn"])

            edge_attr = calculate_edge_features_per_frame(
                trajectories, velocities, edge_indices, frame, config
            )

            graph = Data(
                x=trajectories[:, frame],
                edge_index=edge_indices,
                edge_attr=edge_attr,
            )

            sample_graphs.append(graph)

        graphs.append(sample_graphs)

    return graphs


def center_and_normalize(df: pd.DataFrame, max: float = 1.0) -> pd.DataFrame:
    df["POSITION_X"] = df.groupby("RECORDING")["POSITION_X"].transform(lambda x: x - x.mean())
    df["POSITION_Y"] = df.groupby("RECORDING")["POSITION_Y"].transform(lambda y: y - y.mean())

    df["POSITION_X"] = df.groupby("RECORDING")["POSITION_X"].transform(
        lambda x: (x / np.max(np.abs(x))) * max
    )
    df["POSITION_Y"] = df.groupby("RECORDING")["POSITION_Y"].transform(
        lambda y: (y / np.max(np.abs(y))) * max
    )

    return df


def filter_outliers(df: pd.DataFrame, threshold: float = 4) -> pd.DataFrame:
    """Filters trajectories that contain displacement outliers based on z-score > 4"""

    df = df.sort_values(["RECORDING", "TRACK_ID", "FRAME"]).reset_index(drop=True)

    tracks_x = df.groupby(["RECORDING", "TRACK_ID"])["POSITION_X"]
    tracks_y = df.groupby(["RECORDING", "TRACK_ID"])["POSITION_Y"]

    # Remove frames where positional displacements have z-score > 4
    zscores_x = np.abs(tracks_x.diff().transform(lambda x: (x - x.mean()) / x.std()))
    zscores_y = np.abs(tracks_y.diff().transform(lambda y: (y - y.mean()) / y.std()))

    df.loc[(zscores_x > threshold) | (zscores_y > threshold), ["POSITION_X", "POSITION_Y"]] = (
        np.nan
    )

    return df


def filter_trajectories(
    positions: np.ndarray, features: np.ndarray, threshold: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filters trajectories that contain too many missing frames
    Returns filtered trajectories, features and corresponding recording indices"""

    # Get number of missing frames per trajectory
    missing_frames = np.any(np.isnan(positions), axis=3).sum(axis=2)

    mask = missing_frames < (threshold * positions.shape[2])

    filtered_trajectories = positions[mask]
    filtered_features = features[mask]
    recording_indices = np.array(
        [i for i in range(positions.shape[0]) for j in range(positions.shape[1]) if mask[i, j]]
    )

    return filtered_trajectories, filtered_features, recording_indices


def interpolate_trajectories(data: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values in data of shape (n_trajectories, n_frames, 2)
      using spline interpolation.
    """
    # Create a copy to avoid modifying the input
    result = data.copy()

    # Get array dimensions
    n_trajectories, n_frames, n_coords = data.shape

    # Iterate through all recordings and trajectories
    for traj in range(n_trajectories):
        # Handle each coordinate (x,y) separately
        for coord in range(n_coords):
            # Get current trajectory coordinate
            trajectory = result[traj, :, coord]

            # Check if there are any NaNs to interpolate
            if not np.any(np.isnan(trajectory)):
                continue

            # Get indices of valid (non-NaN) points
            valid_idx = np.where(~np.isnan(trajectory))[0]

            if len(valid_idx) < 4:
                print(f"Trajectory {traj} has too few valid points. Skipping...")
                continue

            # Create time points (using frame indices as time)
            x = valid_idx
            y = trajectory[valid_idx]

            # Fit spline to valid points  k=3 for cubic spline, s=0 for exact interpolation
            spline = UnivariateSpline(x, y, k=3, s=0)

            # Get indices of NaN points
            nan_idx = np.where(np.isnan(trajectory))[0]

            # Only interpolate NaN points that fall within the range of valid points
            mask = (nan_idx >= np.min(valid_idx)) & (nan_idx <= np.max(valid_idx))
            interpolate_idx = nan_idx[mask]

            # Interpolate NaN values
            result[traj, interpolate_idx, coord] = spline(interpolate_idx)

            nan_mask = np.isnan(trajectory)
            # Forward and backward fill NaN values
            if np.any(nan_mask):
                first_valid = np.where(~nan_mask)[0][0]
                # Forward fill
                trajectory[:first_valid] = trajectory[first_valid]
                # Backward fill
                last_valid = np.where(~nan_mask)[0][-1]
                trajectory[last_valid + 1 :] = trajectory[last_valid]

    return result


def align_trajectories(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pads trajectories and aligns them based on track IDs
    to put them in a Numpy array of size (#recordings, #trajectories, #frames, 2)
    Missing frames are filled with NaN"""

    positions = np.full(
        (df["RECORDING"].nunique(), df["TRACK_ID"].nunique(), df["FRAME"].nunique(), 2),
        np.nan,
        dtype=np.float32,
    )
    features = np.full(
        (
            df["RECORDING"].nunique(),
            df["TRACK_ID"].nunique(),
            df["FRAME"].nunique(),
            len(df.columns) - 5,
        ),
        np.nan,
        dtype=np.float32,
    )

    recording_idx = df["RECORDING"].astype("category").cat.codes.values
    track_idx = df["TRACK_ID"].astype("category").cat.codes.values
    frames = df["FRAME"].astype("category").cat.codes.values

    positions[recording_idx, track_idx, frames, :] = df[["POSITION_X", "POSITION_Y"]].values

    features[recording_idx, track_idx, frames, :] = df.drop(
        ["RECORDING", "TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"], axis=1
    ).values

    return (positions, features)


def calculate_local_density(positions: Tensor, config: dict) -> Tensor:
    """
    Compute local cell density using Gaussian kernel density estimation.

    Args:
        positions: Tensor of shape (frames, num_cells, 2) containing cell positions
        config: dict
    Returns:
        densities: Tensor of shape (frames, num_cells, 1) containing local density estimates
    """

    # Compute pairwise distances between all cells, Reshape positions for broadcasting
    pos_i = positions.unsqueeze(2)  # (frames, num_cells, 1, 2)
    pos_j = positions.unsqueeze(1)  # (frames, 1, num_cells, 2)

    # Compute squared distances
    squared_distances = torch.norm(pos_i - pos_j, dim=-1)  # (frames, num_cells, num_cells)

    # Apply Gaussian kernel
    kernel_values = torch.exp(-squared_distances / (2 * config["density_sigma"] ** 2))

    # Sum over all neighboring cells (excluding self-interaction)
    self_mask = torch.eye(positions.shape[1], device=positions.device)[None, :, :]
    kernel_values = kernel_values * (1 - self_mask)

    # Compute density by summing kernel values
    densities = torch.sum(kernel_values, dim=-1, keepdim=True)  # (frames, num_cells, 1)

    # Normalize densities
    densities = densities / torch.max(densities, dim=1, keepdim=True)[0]

    return densities.squeeze(-1)


def calculate_edge_features_per_frame(
    trajectories, velocities, edge_index, frame, config
) -> torch.Tensor:
    """Calculates edge features based on positions of source and target nodes
    args:
        trajectories: Tensor of shape (n_trajectories, n_frames, 2)
        velocities: Tensor of shape (n_trajectories, n_frames, 2)
        edge_index: Tensor of shape (2, n_edges)
        frame: int, current frame"""

    source, target = edge_index

    vel_i, vel_j = velocities[source, frame], velocities[target, frame]

    distances = torch.norm(trajectories[source, frame] - trajectories[target, frame], dim=1)
    # Standardize distances to [0, 1]
    distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Use relative x and y position as edge attribute
    x_rel_pos = trajectories[source, frame, 0] - trajectories[target, frame, 0]
    y_rel_pos = trajectories[source, frame, 1] - trajectories[target, frame, 1]

    edge_attr_list = [torch.tensor([]) for _ in range(len(config["edge_features"]))]
    for feature, index in config["edge_features"].items():
        if feature == "dist":
            edge_attr_list[index] = distances
        elif feature == "delta_x":
            edge_attr_list[index] = x_rel_pos
        elif feature == "delta_y":
            edge_attr_list[index] = y_rel_pos
        elif feature == "delta_vel_x":
            edge_attr_list[index] = vel_i[:, 0] - vel_j[:, 0]
        elif feature == "delta_vel_y":
            edge_attr_list[index] = vel_i[:, 1] - vel_j[:, 1]
        elif feature == "relative_motion":
            direction = (trajectories[source, frame] - trajectories[target, frame]) / (
                distances.unsqueeze(1) + 1e-8
            )
            relative_motion = torch.sum((vel_i - vel_j) * direction, dim=1)
            edge_attr_list[index] = relative_motion
        elif feature == "approaching_speed":
            direction = (trajectories[source, frame] - trajectories[target, frame]) / (
                distances.unsqueeze(1) + 1e-8
            )
            projection_i = torch.sum(vel_i * direction, dim=1)
            projection_j = torch.sum(vel_j * direction, dim=1)

            approaching_speed = -(projection_i - projection_j) / 2
            edge_attr_list[index] = approaching_speed
        elif feature == "relative_density":
            trajs = trajectories[:, frame].unsqueeze(0)
            local_density = calculate_local_density(trajs, config)[0]
            relative_density = local_density[source] - local_density[target]
            edge_attr_list[index] = relative_density

    edge_attr = torch.stack(edge_attr_list, dim=1)

    return edge_attr


def calculate_edge_features(
    trajectories, node_features, edge_index, config, context=None
) -> torch.Tensor:
    """Calculates edge features of whole spatio temporal graph based
    on positions of source and target nodes
    args:
        trajectories: Tensor of shape (batch_size * n_frames * n_trajectories, x)
        node_features: Tensor of shape (batch_size * n_frames * n_trajectories, y)
        edge_index: Tensor of shape (2, n_edges)
        config: dict"""

    source, target = edge_index

    if source.shape[0] == 0 or target.shape[0] == 0:
        return torch.empty((0, len(config["edge_features"])), device=trajectories.device)
    # if trajectories.dim() == 3 or velocities.dim() == 3:
    #     trajectories = trajectories.unsqueeze(0)
    #     velocities = velocities.unsqueeze(0)
    vel_idx = [config["node_features"]["vel_x"], config["node_features"]["vel_y"]]
    vel_i = node_features[:, vel_idx][source]
    vel_j = node_features[:, vel_idx][target]
    trajs = trajectories

    distances = torch.norm(trajs[source] - trajs[target], dim=1)

    # Use relative x and y position as edge attribute
    x_rel_pos = trajs[source, 0] - trajs[target, 0]
    y_rel_pos = trajs[source, 1] - trajs[target, 1]

    # Use gaussian kernel
    exp_distances = torch.exp(-(distances**2) / (2 * config["dist_kernel"] ** 2))

    # Scale positions
    # pos_x = (trajs[:, 0] - config["x_range"][0]) / (
    #     config["x_range"][1] - config["x_range"][0]
    # )
    # pos_y = (trajs[:, 1] - config["y_range"][0]) / (
    #     config["y_range"][1] - config["y_range"][0]
    # )

    if (
        "delta_x_heading" in config["edge_features"]
        and "delta_y_heading" in config["edge_features"]
    ):
        norms = torch.norm(vel_i, dim=1).clip(min=1e-5).unsqueeze(1)
        heading = vel_i / norms  # (heading direction)
        thetas = torch.arctan2(heading[:, 1], heading[:, 0]) - torch.pi / 2
        rot_matrix = torch.zeros(
            vel_i.shape[0],
            2,
            2,
            dtype=vel_i.dtype,
            device=vel_i.device,
        )
        rot_matrix[:, 0, 0] = torch.cos(-thetas)
        rot_matrix[:, 0, 1] = -torch.sin(-thetas)
        rot_matrix[:, 1, 0] = torch.sin(-thetas)
        rot_matrix[:, 1, 1] = torch.cos(-thetas)
        rel_pos_reshaped = torch.stack([x_rel_pos, y_rel_pos], dim=1).unsqueeze(2)
        rotated_rel_pos = torch.bmm(rot_matrix, rel_pos_reshaped).squeeze(2)

    edge_attr = torch.empty(
        (distances.shape[0], len(config["edge_features"])),
        device=trajectories.device,
        dtype=trajectories.dtype,
    )
    for feature, index in config["edge_features"].items():
        if feature == "dist":
            edge_attr[:, index] = exp_distances
        if feature == "euclidean_dist":
            edge_attr[:, index] = distances
        elif feature == "delta_x":
            edge_attr[:, index] = x_rel_pos
        elif feature == "delta_y":
            edge_attr[:, index] = y_rel_pos
        elif feature == "delta_z":
            edge_attr[:, index] = trajs[source, 2] - trajs[target, 2]
        elif feature == "delta_vel_x":
            edge_attr[:, index] = vel_i[:, 0] - vel_j[:, 0]
        elif feature == "delta_vel_y":
            edge_attr[:, index] = vel_i[:, 1] - vel_j[:, 1]
        elif feature == "delta_vel_z":
            vel_z = node_features[:, config["node_features"]["vel_z"]]
            edge_attr[:, index] = vel_z[source] - vel_z[target]
        elif feature == "delta_x_heading":
            edge_attr[:, index] = rotated_rel_pos[:, 0]
        elif feature == "delta_y_heading":
            edge_attr[:, index] = rotated_rel_pos[:, 1]
        elif feature == "relative_turning_angle":
            theta = torch.atan2(vel_j[:, 1], vel_j[:, 0]) - torch.atan2(
                vel_i[:, 1], vel_i[:, 0]
            )
            edge_attr[:, index] = theta / 10
        elif feature == "relative_density":
            density_i = node_features[:, config["node_features"]["local_density"]][source]
            density_j = node_features[:, config["node_features"]["local_density"]][target]
            relative_density = density_i - density_j
            edge_attr[:, index] = relative_density
        elif feature == "charge":
            charge_i = node_features[:, config["node_features"]["charge"]][source]
            charge_j = node_features[:, config["node_features"]["charge"]][target]
            edge_attr[:, index] = torch.where(charge_i == charge_j, 1, -1)
        elif feature == "spring" and context is not None:
            edge_attr[:, index] = context

    return edge_attr


def construct_graphs(
    positions: np.ndarray,
    features: np.ndarray,
    recording_indices: np.ndarray,
    k: int,
    config: dict,
) -> list[list[Data]]:
    """Construct knn graphs from trajectories of shape (n_trajectories, n_frames, 2)
    based on recording indices"""

    graphs = [[] for _ in np.unique(recording_indices)]

    for idx in np.unique(recording_indices):
        # Get all trajectories from the same recording
        trajectories = torch.tensor(positions[recording_indices == idx], dtype=torch.float32)
        rec_features = torch.tensor(features[recording_indices == idx], dtype=torch.float32)

        for frame in range(trajectories.shape[1]):
            edge_indices = knn_graph(trajectories[:, frame], k=k)

            velocities = torch.diff(
                trajectories, dim=1, append=torch.zeros(trajectories.shape[0], 1, 2)
            )

            edge_attr = calculate_edge_features_per_frame(
                trajectories, velocities, edge_indices, frame, config
            )

            graph = Data(
                x=trajectories[:, frame],
                features=rec_features[:, frame],
                edge_index=edge_indices,
                edge_attr=edge_attr,
                pos=trajectories[:, frame, :2],
                time=torch.tensor([frame]),
            )

            graphs[idx].append(graph)

    return graphs


def resize_trajectories(data: np.ndarray, config: dict) -> np.ndarray:
    """Resize trajectories to a fixed number of frames, keeping all data"""

    fraction = config["preprocess_fraction_of_length"]
    n_frames = config["time_window"]
    new_frames = int(n_frames * fraction)
    skip = data.shape[2] // new_frames

    new_data_shape = (data.shape[0] * skip, data.shape[1], new_frames, 2)
    resized_data = np.zeros(new_data_shape, dtype=np.float32)

    for i in range(skip):
        resized_data[i::skip] = data[:, :, i::skip]

    return resized_data


def preprocess(config: dict) -> tuple[np.ndarray, list, np.ndarray]:
    print("Reading data...\n")
    df = read_recordings(
        path=config["data_paths"],
        features=["x", "y"],
    )

    print("Centering and normalizing...\n")
    df = center_and_normalize(df, max=config["max_x_and_y"])

    print("Filtering outliers...\n")
    df = filter_outliers(df, config["preprocess_outlier_zscore_threshold"])

    print("Aligning trajectories...\n")
    positions, features = align_trajectories(df)

    print("Filtering trajectories...\n")
    # Transforms data to shape (n_trajectories, n_frames, 2)
    trajectories, features, recording_indices = filter_trajectories(
        positions, features, threshold=config["preprocess_nans_in_trajectory_threshold"]
    )

    # Cut off first and last frames as they contain many NaNs
    frames_to_cut = (trajectories.shape[1] - config["time_window"]) // 2
    trajectories = trajectories[:, frames_to_cut : trajectories.shape[1] - frames_to_cut, :]
    features = features[:, frames_to_cut : features.shape[1] - frames_to_cut, :]

    nan_traj = trajectories[np.isnan(trajectories).any(axis=(1, 2))]
    print(f"Trajectories containing NaNs: {nan_traj.shape[0]} / {trajectories.shape[0]}\n")

    print("Interpolating trajectories...\n")
    final_trajectories = interpolate_trajectories(trajectories)

    # Forward fill features
    for i in range(features.shape[2]):
        features[:, :, i] = pd.DataFrame(features[:, :, i]).fillna(method="ffill").values  # type: ignore

    print(f"Trajectories shape: {final_trajectories.shape}\n")
    print(f"Features shape: {features.shape}\n")

    print("Constructing graphs...\n")
    graphs = construct_graphs(
        final_trajectories,
        features,
        recording_indices,
        k=config["knn"],
        config=config,
    )

    print(f"Saving {len(graphs)} graphs...\n")
    np.savez_compressed(
        config["trajectories_path"],
        trajectories=final_trajectories,
        recording_indices=recording_indices,
    )

    torch.save({"graphs": graphs}, config["graphs_path"])

    return final_trajectories, graphs, recording_indices


def preprocess_boids(config: dict) -> np.ndarray:
    print("Reading boids data...\n")
    boids = read_recordings_boids(config["data_paths"])

    print("Handling boundaries...\n")
    boids = handle_boundaries(boids)

    print("Centering and scaling boids data...\n")
    boids = center_and_scale_boids(boids, max=config["max_x_and_y"])

    print("Resizing...\n")
    boids = resize_trajectories(boids, config)

    print("Constructing boids graphs...\n")
    # graphs = construct_boids_graphs(boids, config)

    np.savez_compressed(config["save_path"], trajectories=boids)
    # torch.save(graphs, config["preprocess"]["graphs_path"])

    return boids


def preprocess_pedestrians(config: dict) -> list[np.ndarray]:
    print("Reading pedestrians data...\n")
    if "Eindhoven" in config["data_paths"][0]:
        pedestrians = read_and_process_pedestrians_eindhoven(config)
    elif "SDD" in config["data_paths"][0]:
        pedestrians = read_and_process_SDD(config)
    else:
        pedestrians = read_and_process_pedestrians_ETH(config)

    print("Forward and backwards filling NaNs...\n")
    for sample in pedestrians:
        mask = np.isnan(sample)
        if mask.any():
            for t in range(sample.shape[0]):
                for f in range(sample.shape[1] - 1):
                    sample[t, f + 1][mask[t, f + 1]] = sample[t, f][mask[t, f + 1]]

            for t in range(sample.shape[0]):
                for f in range(sample.shape[1] - 2, -1, -1):
                    sample[t, f][mask[t, f]] = sample[t, f + 1][mask[t, f]]

    print("Saving pedestrians data...\n")
    np.save(
        config["save_path"],
        np.array(pedestrians, dtype=object),
        allow_pickle=True,
    )

    return pedestrians
