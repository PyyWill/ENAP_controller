import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

def normalize_feature(feat, mean, std):
    """Apply Z-score normalization."""
    return (feat - mean) / (std + 1e-8)

def build_feature_indexes_from_pmm(pmm, norm_stats, episodes=None):
    """
    Build normalized feature index (rgb + pos) using PMM edge_cache.
    State index comes from PMM symbol x, not argmax(state), so it supports
    remapped clusters (e.g. split c2 -> c2_0/c2_1).

    Feature source priority: pmm.raw_features (stored in pkl) > episodes arg.
    """
    feat_src = getattr(pmm, 'raw_features', None) or episodes
    assert feat_src is not None, (
        "No feature source: pmm.raw_features is None and no episodes provided. "
        "Re-train/re-save PMM with raw_episodes to embed features in the pkl.")

    all_features, all_state_indices = [], []
    for (q, x, _), step_list in pmm._edge_cache.items():
        for epi, t in step_list:
            step = feat_src[epi][t]
            rgb, pos = step["rgb_feature"], step["pos_feature"]

            rgb_n = normalize_feature(rgb, norm_stats["rgb_mean"], norm_stats["rgb_std"])
            pos_n = normalize_feature(pos, norm_stats["pos_mean"], norm_stats["pos_std"])

            all_features.append(np.concatenate([rgb_n, pos_n]))
            all_state_indices.append(int(x))

    return (np.array(all_features), np.array(all_state_indices))

def load_trajectories(pkl_path):
    """Load trajectory data and cluster centers"""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    episodes = payload["episodes"]
    cluster_centers = payload.get("cluster_centers", None)
    
    print(f"Loading {len(episodes)} episodes")
    if cluster_centers is not None:
        print(f"Loaded cluster_centers with shape {cluster_centers.shape}")
    # print keys in episodes
    
    return episodes, cluster_centers