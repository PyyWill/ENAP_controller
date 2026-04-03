#!/usr/bin/env python3
"""PMM + Residual policy evaluation on PegInsertionSide-v1."""
import os
import random
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_agent import PMMResidualAgent, ResidualMLP
from agent.pmm_class import PMM
from agent.utils import build_feature_indexes_from_pmm, load_trajectories


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _load_ppo_feature_net(agent, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("agent_state_dict", ckpt)
    feat = {k[len("feature_net."):]: v for k, v in state.items() if k.startswith("feature_net.")}
    assert feat, f"No feature_net weights in {ckpt_path}"
    agent.feature_net.load_state_dict(feat, strict=True)
    for p in agent.feature_net.parameters():
        p.requires_grad = False
    agent.feature_net.eval()


@dataclass
class EvalArgs:
    num_episodes: int = 64
    num_envs: int = 64
    env_id: str = "PegInsertionSide-v1"
    residual_weights: str = "results/checkpoints/residual_net.pt"
    traj_pkl: str = "data/episodes_with_states.pkl"
    pmm_pkl: str = "results/pmm_residual_training.pkl"
    encoder_weights: str = "/home/yiyuanp/Project/ENAP_public/results/checkpoints/oracle_encoder.pt"
    seed: int = 1
    max_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = tyro.cli(EvalArgs)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    checkpoint = torch.load(args.residual_weights, map_location=args.device, weights_only=False)
    norm_stats = checkpoint["norm_stats"]
    episodes, cluster_centers = load_trajectories(args.traj_pkl)

    pmm = PMM(use_tqdm=True, max_inner_iters=1000)
    pmm.load_pmm(args.pmm_pkl)
    print(f"PMM: {len(pmm.pmm['Q'])} nodes, {len(pmm.pmm['delta'])} edges")

    feature_indexes = build_feature_indexes_from_pmm(pmm, norm_stats, episodes=episodes)
    a_dim = len(episodes[0][0]["action"])
    feat_dim = len(norm_stats["rgb_mean"]) + len(norm_stats["pos_mean"])

    residual_net = ResidualMLP(feat_dim, a_dim, hidden=256).to(args.device)
    residual_net.load_state_dict(checkpoint["model"])
    residual_net.eval()

    # Environment
    env_kwargs = dict(obs_mode="rgb", render_mode="rgb_array", sim_backend="physx_cuda")
    reconfig_freq = 1 if args.num_envs == 1 else 0
    env = gym.make(args.env_id, num_envs=args.num_envs,
                   reconfiguration_freq=reconfig_freq, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=False, record_metrics=True)

    # Agent
    obs, _ = env.reset(seed=args.seed)
    agent = PMMResidualAgent(
        envs=env, sample_obs=obs, pmm=pmm, residual_net=residual_net,
        cluster_centers=cluster_centers, device=args.device,
        feature_indexes=feature_indexes, norm_stats=norm_stats,
    )
    agent.to(args.device)
    _load_ppo_feature_net(agent, args.encoder_weights, args.device)
    agent.eval()

    # Eval loop
    success_count = 0
    episode_index = 0
    total_batches = (args.num_episodes + args.num_envs - 1) // args.num_envs

    for batch_idx in range(total_batches):
        obs, _ = env.reset(seed=args.seed + batch_idx)
        done = np.zeros(args.num_envs, dtype=bool)
        episode_success = np.zeros(args.num_envs, dtype=bool)
        agent.reset_pmm_state()

        for _ in range(args.max_steps):
            obs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu()

            obs, reward, terminations, truncations, infos = env.step(action)
            done_step = _to_numpy(terminations).astype(bool) | _to_numpy(truncations).astype(bool)

            if "final_info" in infos:
                final_mask = _to_numpy(infos.get("_final_info", np.ones(args.num_envs)))
                final_info = infos["final_info"]
                for idx in np.where(done_step & ~done)[0]:
                    if not bool(final_mask[idx]):
                        continue
                    info_i = final_info[idx] if isinstance(final_info, (list, tuple)) else final_info
                    for key in ("success",):
                        if key in info_i:
                            val = info_i[key]
                            val = val[idx].item() if hasattr(val, '__len__') and len(val) > 1 else (val.item() if hasattr(val, 'item') else val)
                            episode_success[idx] = bool(val)
                            break

            done = done | done_step
            if np.all(done):
                break

        for env_idx in range(args.num_envs):
            episode_index += 1
            if episode_index > args.num_episodes:
                break
            if episode_success[env_idx]:
                success_count += 1

    success_rate = 100.0 * success_count / max(args.num_episodes, 1)
    print(f"\n{args.env_id} | {args.num_episodes} eps | "
          f"Success: {success_count}/{args.num_episodes} ({success_rate:.1f}%)")

    env.close()


if __name__ == "__main__":
    main()
