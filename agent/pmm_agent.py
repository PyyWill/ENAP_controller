#!/usr/bin/env python3
"""Utility functions and classes for PMM training and evaluation"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from agent.utils import NatureCNN

class PMMResidualAgent(nn.Module):
    """Agent combining PMM + Residual Network with PPO"""
    def __init__(self, envs, sample_obs, pmm, residual_net, cluster_centers, device, feature_indexes=None, norm_stats=None, encoder_ckpt=None):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        self.pmm = pmm
        self.residual_net = residual_net
        self.cluster_centers = cluster_centers
        self.device = device
        self.norm_stats = norm_stats
        
        self.feat_dim = len(norm_stats["rgb_mean"]) + len(norm_stats["pos_mean"])
        self.num_envs = envs.num_envs
        self.current_q = torch.zeros(envs.num_envs, dtype=torch.long, device=device)
        self.feature_indexes = feature_indexes
        self.feat_array_t = None
        self.state_indices_idx = None

        assert pmm.rnn is not None, "PMM must have a trained RNN"
        rnn_h_dim = pmm.rnn.h_dim
        self.rnn_h = np.zeros((envs.num_envs, rnn_h_dim), dtype=np.float32)
        
        if self.feature_indexes is not None:
            feat_array, state_indices_idx = self.feature_indexes
            self.feat_array_t = torch.from_numpy(feat_array).float().to(self.device)
            self.state_indices_idx = state_indices_idx
        
        self.feature_net.eval()
        for p in self.feature_net.parameters():
            p.requires_grad = False
    
    def reset_pmm_state(self, env_indices=None):
        if env_indices is None:
            self.current_q.zero_()
            self.rnn_h[:] = 0
        else:
            env_indices = torch.as_tensor(env_indices, device=self.device).long()
            env_indices = torch.clamp(env_indices, 0, self.num_envs - 1)
            self.current_q[env_indices] = 0
            self.rnn_h[env_indices.cpu().numpy()] = 0
    
    def ensure_validity(self, dists, state_idx_batch, current_q_batch, state_indices_idx):
        """Ensure (curr_q, s_idx) is in the domain of PMM""" 
        batch_size = len(state_idx_batch)
        state_idx_batch = state_idx_batch.copy()
        state_indices_array = state_indices_idx
        
        for i in range(batch_size):
            curr_q = current_q_batch[i]
            s_idx = state_idx_batch[i]
            
            if (curr_q, s_idx) not in self.pmm.pmm['delta']:
                valid_state_indices = [s_idx_valid for (q, s_idx_valid) in self.pmm.pmm['delta'].keys() if q == curr_q]
                if valid_state_indices:
                    valid_mask = np.isin(state_indices_array, valid_state_indices)
                    valid_dists = dists[i].clone()
                    valid_dists[~torch.from_numpy(valid_mask).to(self.device)] = float('inf')
                    nearest_idx = torch.argmin(valid_dists).item()
                    state_idx_batch[i] = state_indices_array[nearest_idx]
        return state_idx_batch
    
    def _disambiguate(self, trans_probs, z_t):
        """Pick next node: single edge → follow; multiple → cosine with rep_embeddings."""
        if len(trans_probs) == 1:
            return next(iter(trans_probs))
        rep_embeddings = self.pmm.pmm.get('rep_embeddings', [])
        best_q, best_sim = None, -2.0
        if z_t is not None and rep_embeddings:
            z_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
            for q_cand in trans_probs:
                if q_cand < len(rep_embeddings):
                    z_rep = rep_embeddings[q_cand]
                    sim = np.dot(z_norm, z_rep / (np.linalg.norm(z_rep) + 1e-12))
                    if sim > best_sim:
                        best_sim = sim
                        best_q = q_cand
        if best_q is None:
            raise RuntimeError("PMM multi-edge disambiguation failed")
        return best_q

    def get_pmm_action(self, obs, env_indices=None, external_q=None, delta_pos=None):
        """Forward pass: features → PMM action list → Residual → RNN step → node transition."""
        batch_size = obs["rgb"].shape[0]

        rgb_raw = self.feature_net.extractors["rgb"](obs["rgb"].float().permute(0, 3, 1, 2) / 255.0)
        assert "state" in obs, "obs must contain 'state'"
        assert "state" in self.feature_net.extractors, "feature_net must have 'state' extractor"
        pos_raw = self.feature_net.extractors["state"](obs["state"])

        rgb_mean = torch.from_numpy(self.norm_stats['rgb_mean']).float().to(self.device)
        rgb_std = torch.from_numpy(self.norm_stats['rgb_std']).float().to(self.device)
        rgb_features = (rgb_raw - rgb_mean) / (rgb_std + 1e-8)

        pos_mean = torch.from_numpy(self.norm_stats['pos_mean']).float().to(self.device)
        pos_std = torch.from_numpy(self.norm_stats['pos_std']).float().to(self.device)
        pos_features = (pos_raw - pos_mean) / (pos_std + 1e-8)

        concat_feat_matching = torch.cat([rgb_features, pos_features], dim=-1)
        dists = torch.cdist(concat_feat_matching, self.feat_array_t)
        nearest_indices = torch.argmin(dists, dim=1).cpu().numpy()
        state_idx_batch = self.state_indices_idx[np.clip(nearest_indices, 0, len(self.state_indices_idx) - 1)]

        if external_q is not None:
            current_q_batch = external_q.cpu().numpy()
        else:
            if env_indices is None:
                env_indices = torch.arange(batch_size, device=self.device)
            env_indices = torch.as_tensor(env_indices, device=self.device).long()
            env_indices = torch.clamp(env_indices, 0, self.num_envs - 1)
            current_q_batch = self.current_q[env_indices].cpu().numpy()

        state_idx_batch = self.ensure_validity(dists, state_idx_batch, current_q_batch, self.state_indices_idx)

        action_lists, cluster_centers_list = [], []
        for i in range(batch_size):
            curr_q = current_q_batch[i]
            s_idx = state_idx_batch[i]

            with torch.no_grad():
                a_list = self.pmm.predict_list(curr_q, s_idx)

            assert a_list is not None, f"predict_list returned None for (q={curr_q}, x={s_idx})"
            action_lists.append(a_list)

            assert self.cluster_centers is not None, "cluster_centers must be set"
            assert 0 <= s_idx < len(self.cluster_centers), f"s_idx={s_idx} out of range [0, {len(self.cluster_centers)})"
            cc = self.cluster_centers[s_idx]
            cluster_centers_list.append(cc)

        # Batched residual forward
        concat_feat_residual = torch.cat([rgb_features, pos_features], dim=-1)
        cluster_centers_t = torch.from_numpy(np.array(cluster_centers_list)).float().to(self.device)
        action_output = self.residual_net(action_lists, cluster_centers_t, concat_feat_residual)

        # Per-env: RNN step with predicted action, then embedding-based node transition
        if external_q is None:
            action_np = action_output.detach().cpu().numpy()
            env_idx_np = env_indices.cpu().numpy()
            for i in range(batch_size):
                ei = int(env_idx_np[i])
                s_idx = state_idx_batch[i]
                h_new = self.pmm.rnn.forward_step(action_np[i], s_idx, self.rnn_h[ei])
                self.rnn_h[ei] = h_new

                trans_probs = self.pmm.pmm['delta'].get((current_q_batch[i], s_idx), {})
                if trans_probs:
                    self.current_q[ei] = self._disambiguate(trans_probs, h_new)

        return action_output

    def get_action(self, x, deterministic=False):
        action_mean = self.get_pmm_action(x)
        if deterministic:
            return action_mean
        
        # Exploration logic
        logstd = torch.clamp(self.residual_net.logstd, min=-2.0, max=2.0).expand_as(action_mean)
        std = torch.exp(logstd)
        probs = Normal(action_mean, std)
        return probs.sample()

    def get_action_and_logprob(self, x, action=None, env_indices=None, external_q=None, delta_pos=None):
        action_mean = self.get_pmm_action(x, env_indices=env_indices, external_q=external_q, delta_pos=delta_pos)
        
        logstd = torch.clamp(self.residual_net.logstd, min=-2.0, max=2.0).expand_as(action_mean)
        std = torch.exp(logstd)
        probs = Normal(action_mean, std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class ResidualMLP(nn.Module):
    """Mean of PMM action list as prior + residual MLP (no clamp)."""

    def __init__(self, feat_dim, a_dim, hidden=512):
        super().__init__()
        self.a_dim = a_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim + a_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim),
        )
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.net[-1].bias)

        self.logstd = nn.Parameter(torch.ones(1, a_dim) * -0.5)

    def _action_base(self, action_list, device):
        """Per-batch mean over variable-length action lists -> (B, a_dim)."""
        B = len(action_list)
        lengths = [np.asarray(a, dtype=np.float32).reshape(-1, self.a_dim).shape[0] for a in action_list]
        max_len = max(lengths) if lengths else 0
        padded = torch.zeros(B, max_len, self.a_dim, device=device)
        mask = torch.zeros(B, max_len, device=device)
        for i, a in enumerate(action_list):
            arr = torch.as_tensor(np.asarray(a, dtype=np.float32), device=device).reshape(-1, self.a_dim)
            n = arr.shape[0]
            padded[i, :n] = arr
            mask[i, :n] = 1.0
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (padded * mask.unsqueeze(-1)).sum(dim=1) / count

    def forward(self, action_list, cluster_center, concat_feat):
        """
        action_list : list[B] of arrays, each (N_i, a_dim)  — variable length
        cluster_center : (B, feat_dim)
        concat_feat    : (B, feat_dim)
        Returns        : (B, a_dim)
        """
        delta = concat_feat - cluster_center
        action_base = self._action_base(action_list, concat_feat.device)
        return self.net(torch.cat([delta, action_base], dim=-1)) + action_base

    def get_action_mean(self, action_list, cluster_center, concat_feat):
        return self.forward(action_list, cluster_center, concat_feat)

    def sample_action(self, action_list, cluster_center, concat_feat):
        action_mean = self.forward(action_list, cluster_center, concat_feat)
        logstd = torch.clamp(self.logstd, min=-2.0, max=2.0).expand_as(action_mean)
        return Normal(action_mean, torch.exp(logstd)).sample()

    def get_log_prob(self, action_list, cluster_center, concat_feat, action):
        action_mean = self.forward(action_list, cluster_center, concat_feat)
        logstd = torch.clamp(self.logstd, min=-2.0, max=2.0).expand_as(action_mean)
        return Normal(action_mean, torch.exp(logstd)).log_prob(action).sum(dim=-1)