#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTERING_DIR = os.path.join(SCRIPT_DIR, "clustering")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_features(pkl_path):
    """Extract RGB and position features from ManiSkill trajectory pkl file."""
    data = load_pickle(pkl_path)
    episodes = data['episodes'] if isinstance(data, dict) else data
    metadata = data.get('metadata', None) if isinstance(data, dict) else None

    all_rgb_features = []
    all_pos_features = []
    all_rgb_images = []

    for episode in tqdm(episodes, desc="Extracting features"):
        for step in episode:
            if 'rgb_feature' in step and step['rgb_feature'] is not None \
               and 'pos_feature' in step and step['pos_feature'] is not None:
                rgb_feat = step['rgb_feature']
                if rgb_feat.ndim > 1:
                    rgb_feat = rgb_feat.reshape(-1)

                pos_feat = step['pos_feature']
                if pos_feat.ndim > 1:
                    pos_feat = pos_feat.reshape(-1)

                all_rgb_features.append(rgb_feat)
                all_pos_features.append(pos_feat)

                if 'rgb' in step:
                    rgb_img = step['rgb']
                    if rgb_img.shape[2] == 6:
                        rgb_img = rgb_img[:, :, :3]
                    all_rgb_images.append(rgb_img)

    rgb_features = np.array(all_rgb_features)
    pos_features = np.array(all_pos_features)
    rgb_images = np.array(all_rgb_images)

    print(f"Read {len(all_rgb_features)} features")
    print(f"RGB feature shape: {rgb_features.shape}")
    print(f"POS feature shape: {pos_features.shape}")
    print(f"RGB images shape: {rgb_images.shape}")

    return rgb_features, pos_features, rgb_images, episodes, metadata


def hdbscan_cluster(X):
    n = X.shape[0]
    min_cluster_size = max(10, int(round(0.035 * n)))
    min_samples = max(2, int(round(0.25 * min_cluster_size)))
    print(f"Min cluster size: {min_cluster_size}, Min samples: {min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.05,
        metric='euclidean',
    )
    return clusterer.fit_predict(X)


def visualize(Z, labels, out_png):
    """PCA 2D visualization of clustering results."""
    Y = PCA(n_components=2, random_state=0).fit_transform(Z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=140)

    uniq = np.unique(labels)
    uniq_sorted = sorted(uniq[uniq != -1])
    has_noise = -1 in uniq
    if has_noise:
        uniq_sorted.append(-1)

    n_clusters = len(uniq_sorted) - (1 if has_noise else 0)

    if n_clusters <= 10:
        colors = np.vstack([
            plt.cm.Set2(np.linspace(0, 1, 8)),
            plt.cm.Dark2(np.linspace(0, 1, 8)),
        ])[:n_clusters]
    elif n_clusters <= 20:
        colors = np.vstack([
            plt.cm.Set2(np.linspace(0, 1, 8)),
            plt.cm.Dark2(np.linspace(0, 1, 8)),
            plt.cm.tab10(np.linspace(0, 1, 10)),
        ])[:n_clusters]
    else:
        colors = plt.cm.turbo(np.linspace(0.1, 0.9, n_clusters))

    for i, lab in enumerate(uniq_sorted):
        mask = labels == lab
        if lab == -1:
            ax1.scatter(Y[mask, 0], Y[mask, 1], s=8, color="#666666",
                        label=f"noise (-1, n={np.sum(mask)})", alpha=0.01,
                        edgecolors='none', marker='x')
        else:
            color = colors[i % len(colors)]
            ax1.scatter(Y[mask, 0], Y[mask, 1], s=15, color=color,
                        label=f"c{lab} (n={np.sum(mask)})", alpha=0.9,
                        edgecolors='gray', linewidths=0.8)

    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(markerscale=2, fontsize=7, ncol=2, frameon=False, loc='upper right')
    ax1.set_title(f"All clusters (PCA) - {n_clusters} clusters", fontsize=12)

    if has_noise:
        cluster_mask = labels != -1
        Y_clusters = Y[cluster_mask]
        labels_clusters = labels[cluster_mask]
        for i, lab in enumerate(uniq_sorted[:-1]):
            mask = labels_clusters == lab
            color = colors[i % len(colors)]
            ax2.scatter(Y_clusters[mask, 0], Y_clusters[mask, 1], s=15,
                        color=color, label=f"c{lab} (n={np.sum(mask)})",
                        alpha=0.9, edgecolors='gray', linewidths=0.8)
    else:
        for i, lab in enumerate(uniq_sorted):
            mask = labels == lab
            color = colors[i % len(colors)]
            ax2.scatter(Y[mask, 0], Y[mask, 1], s=15, color=color,
                        label=f"c{lab} (n={np.sum(mask)})", alpha=0.9,
                        edgecolors='gray', linewidths=0.8)

    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.legend(markerscale=2, fontsize=7, ncol=2, frameon=False, loc='upper right')
    ax2.set_title("Clusters only (PCA)", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def save_image_grid(images, title, out_path, color_rgb, grid_size):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif grid_size == 2:
        axes = axes.reshape(2, 2)
    axes_flat = axes.flatten()
    fig.suptitle(title, fontsize=14)
    for idx, img in enumerate(images):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        axes_flat[idx].imshow(img)
        axes_flat[idx].set_title(f"#{idx+1}", fontsize=8)
        axes_flat[idx].axis('off')
        for spine in axes_flat[idx].spines.values():
            spine.set_edgecolor(color_rgb)
            spine.set_linewidth(3)
    for idx in range(len(images), grid_size * grid_size):
        axes_flat[idx].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def save_cluster_images(rgb_data, labels, out_dir, n_samples_per_cluster=16):
    """Save representative RGB images for each cluster."""
    cluster_labels = np.unique(labels)
    cluster_labels = cluster_labels[cluster_labels != -1]
    if len(cluster_labels) == 0:
        return

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(cluster_labels))))

    for i, label in enumerate(tqdm(cluster_labels, desc="Saving cluster images")):
        mask = labels == label
        cluster_rgb = rgb_data[mask]
        n_cluster = len(cluster_rgb)

        n_samples = min(n_samples_per_cluster, n_cluster)
        selected_indices = (np.arange(n_cluster) if n_samples == n_cluster
                            else np.linspace(0, n_cluster - 1, n_samples, dtype=int))

        grid_size = int(np.ceil(np.sqrt(n_samples)))
        color_rgb = colors[i % len(colors)][:3]

        save_image_grid(cluster_rgb[selected_indices],
                        f"Cluster {label} - RGB (n={n_cluster})",
                        os.path.join(out_dir, f"cluster_{label}_rgb.png"),
                        color_rgb, grid_size)

    if -1 in labels:
        noise_rgb = rgb_data[labels == -1]
        if len(noise_rgb) > 0:
            n_samples = min(n_samples_per_cluster, len(noise_rgb))
            selected_indices = (np.arange(len(noise_rgb)) if n_samples == len(noise_rgb)
                                else np.linspace(0, len(noise_rgb) - 1, n_samples, dtype=int))
            grid_size = int(np.ceil(np.sqrt(n_samples)))
            save_image_grid(noise_rgb[selected_indices],
                            f"Noise points (-1) - RGB (n={len(noise_rgb)})",
                            os.path.join(out_dir, "cluster_noise_rgb.png"),
                            [0.5, 0.5, 0.5], grid_size)


def compute_cluster_centers(Z, labels):
    """Compute cluster centroids in the feature space."""
    unique_labels = sorted(np.unique(labels))
    cluster_centers = []
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            cluster_centers.append(Z[mask].mean(axis=0))
    return np.array(cluster_centers), unique_labels


def add_states_to_episodes(episodes, labels):
    """Add one-hot state vectors to each step based on cluster labels."""
    labels_unique = sorted(np.unique(labels))
    label_to_idx = {label: idx for idx, label in enumerate(labels_unique)}
    total_states = len(labels_unique)
    label_idx = 0
    for episode in tqdm(episodes, desc="Adding states"):
        for step in episode:
            if 'rgb_feature' in step and step['rgb_feature'] is not None \
               and 'pos_feature' in step and step['pos_feature'] is not None:
                state_index = label_to_idx.get(labels[label_idx], 0)
                state = np.zeros(total_states, dtype=np.float32)
                if 0 <= state_index < total_states:
                    state[state_index] = 1.0
                step['state'] = state
                label_idx += 1
    return episodes


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="trajectories.pkl")
    ap.add_argument("--n-samples", type=int, default=16,
                    help="Number of sample images per cluster")
    ap.add_argument("--output-pkl", type=str, default="",
                    help="Path to save episodes_with_states pkl "
                         "(default: <script_dir>/episodes_with_states.pkl)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(CLUSTERING_DIR, exist_ok=True)

    rgb_features, pos_features, rgb_images, episodes, metadata = extract_features(args.pkl)
    print(rgb_features.shape, pos_features.shape)

    print("\nZ-score → RGB PCA(10D) → HDBSCAN (centers use RGB+POS concat)...")
    rgb_features_norm = (rgb_features - rgb_features.mean(axis=0)) / (rgb_features.std(axis=0) + 1e-8)
    pos_features_norm = (pos_features - pos_features.mean(axis=0)) / (pos_features.std(axis=0) + 1e-8)

    combined_features = np.concatenate([rgb_features_norm, pos_features_norm], axis=1)
    features_pca = PCA(n_components=10, random_state=0).fit_transform(rgb_features_norm)
    labels = hdbscan_cluster(features_pca)

    np.save(os.path.join(CLUSTERING_DIR, "maniskill_cluster_labels.npy"), labels)
    visualize(features_pca, labels, os.path.join(CLUSTERING_DIR, "maniskill_clusters_hdbscan.png"))
    save_cluster_images(rgb_images, labels, CLUSTERING_DIR, n_samples_per_cluster=args.n_samples)

    uniq_labels = np.unique(labels)
    n_clusters = len(uniq_labels[uniq_labels != -1])
    total_states = len(uniq_labels)
    print(f"\nClustering: {len(labels)} samples, {n_clusters} clusters")
    print(f"Total states: {total_states}")

    print("\nCluster sample counts:")
    for label in sorted(uniq_labels):
        count = np.sum(labels == label)
        name = f"Noise (-1)" if label == -1 else f"Cluster {label}"
        print(f"  {name}: {count} samples")

    episodes_with_states = add_states_to_episodes(episodes, labels)

    cluster_centers, center_labels = compute_cluster_centers(combined_features, labels)
    print(f"Computed {len(cluster_centers)} cluster centers (dim={cluster_centers.shape[1]})")

    output_pkl_path = args.output_pkl or os.path.join(SCRIPT_DIR, "episodes_with_states.pkl")
    output_data = {
        "episodes": episodes_with_states,
        "num_clusters": n_clusters,
        "num_states": total_states,
        "cluster_centers": cluster_centers,
        "cluster_labels": center_labels,
    }
    if metadata is not None:
        output_data["metadata"] = metadata

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nSaved {len(episodes_with_states)} episodes with states to {output_pkl_path}")
    print(f"Total states: {total_states}")
    print(f"Saved {len(cluster_centers)} cluster centers (including noise center if present)")

    if len(episodes_with_states) > 0:
        first_episode = episodes_with_states[0]
        state_indices = [int(np.argmax(step['state'])) for step in first_episode if 'state' in step]

        if state_indices:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(state_indices, marker='o', markersize=3, linewidth=1.5, alpha=0.7, color='blue')
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('State Index', fontsize=12)
            ax.set_title('First Episode State Transitions Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)

            tick_step = max(1, len(state_indices) // 20)
            ax.set_xticks(range(0, len(state_indices), tick_step))

            y_min, y_max = min(state_indices), max(state_indices)
            y_range = y_max - y_min if y_max != y_min else 1
            for i in range(1, len(state_indices)):
                if state_indices[i] != state_indices[i - 1]:
                    ax.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(i, y_max - 0.1 * y_range, f'step={i}',
                            rotation=90, ha='right', va='top', fontsize=8, color='red')

            plt.tight_layout()
            timeline_path = os.path.join(CLUSTERING_DIR, "first_episode_states_timeline.png")
            plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nSaved first episode state timeline to {timeline_path}")
