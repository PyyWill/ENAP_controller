import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_class import PMM
from agent.pmm_agent import ResidualMLP
from agent.utils import build_feature_indexes_from_pmm, normalize_feature, load_trajectories


# ---------------------------------------------------------------------------
# Node transition (consistent with pmm_class.replay_assign)
# ---------------------------------------------------------------------------
def compute_episode_rnn_embeddings(episode, pmm):
    """Run the PMM's RNN over an episode and return hidden states for every step."""
    actions = np.stack([step['action'] for step in episode], axis=0)
    state_indices = np.array([int(np.argmax(step['state'])) for step in episode])
    return pmm.rnn.forward_trajectory(actions, state_indices)


def update_pmm_node(pmm, current_q, x, z_t):
    """Transition using embedding-based disambiguation (mirrors replay_assign).

    - 0 outgoing edges  -> self-loop
    - 1 outgoing edge   -> follow it
    - multiple edges    -> pick destination whose rep_embedding is closest to z_t
    """
    dist = pmm.pmm['delta'].get((current_q, x))
    if dist is None:
        return current_q
    if len(dist) == 1:
        return next(iter(dist))

    rep_embeddings = pmm.pmm.get('rep_embeddings', [])
    best_q, best_sim = None, -2.0
    if z_t is not None and rep_embeddings:
        z_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
        for q_cand in dist:
            if q_cand < len(rep_embeddings):
                z_rep = rep_embeddings[q_cand]
                sim = np.dot(z_norm, z_rep / (np.linalg.norm(z_rep) + 1e-12))
                if sim > best_sim:
                    best_sim = sim
                    best_q = q_cand
    if best_q is None:
        raise RuntimeError("PMM multi-edge disambiguation failed")
    return best_q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_valid_state_idx(current_q, state_idx, concat_feat, pmm, feature_indexes, dists=None, dist_idx=None):
    """Find a valid state_idx for current_q."""
    if (current_q, state_idx) in pmm.pmm['delta']:
        return state_idx

    feat_array, state_indices_array = feature_indexes
    valid_state_indices = [s_idx for (q, s_idx) in pmm.pmm['delta'].keys() if q == current_q]
    if not valid_state_indices:
        return state_idx

    if dists is not None and dist_idx is not None:
        valid_mask = np.isin(state_indices_array, valid_state_indices)
        valid_dists = dists[dist_idx].clone()
        valid_dists[~torch.from_numpy(valid_mask).to(valid_dists.device)] = float('inf')
        nearest_idx = torch.argmin(valid_dists).item()
    else:
        distances = np.linalg.norm(feat_array - concat_feat, axis=1)
        distances[~np.isin(state_indices_array, valid_state_indices)] = np.inf
        nearest_idx = np.argmin(distances)

    return state_indices_array[nearest_idx]


def compute_normalization_stats(episodes):
    all_rgb, all_pos = [], []
    for ep in episodes:
        for step in ep:
            all_rgb.append(step['rgb_feature'])
            all_pos.append(step['pos_feature'])
    all_rgb = np.array(all_rgb)
    all_pos = np.array(all_pos)
    return {
        'rgb_mean': all_rgb.mean(axis=0), 'rgb_std': all_rgb.std(axis=0),
        'pos_mean': all_pos.mean(axis=0), 'pos_std': all_pos.std(axis=0),
    }


def collect_episode_data(episodes, pmm, norm_stats):
    """Preprocess episodes into flat list; node transitions use RNN embedding."""
    all_data = []
    for episode in episodes:
        if not episode:
            continue

        h_seq = compute_episode_rnn_embeddings(episode, pmm)
        current_q = 0

        for step_idx, step in enumerate(episode):
            rgb_norm = normalize_feature(step['rgb_feature'], norm_stats['rgb_mean'], norm_stats['rgb_std'])
            pos_norm = normalize_feature(step['pos_feature'], norm_stats['pos_mean'], norm_stats['pos_std'])
            concat_feat = np.concatenate([rgb_norm, pos_norm]).astype(np.float32)
            state_idx = int(np.argmax(step['state']))

            all_data.append({
                'current_q': current_q,
                'state_idx': state_idx,
                'action_true': step['action'].astype(np.float32),
                'concat_feat': concat_feat,
            })

            current_q = update_pmm_node(pmm, current_q, state_idx, h_seq[step_idx])

    return all_data


def process_batch(batch_data, pmm, feat_array_t, state_indices_array, node_to_mask_t, cluster_centers_t, a_dim, device):
    """Batch processing using pre-pushed GPU caches."""
    concat_feat_batch = np.array([d['concat_feat'] for d in batch_data])
    concat_feat_t = torch.from_numpy(concat_feat_batch).to(device)
    dists = torch.cdist(concat_feat_t, feat_array_t, p=2)

    action_lists_batch, cluster_centers_batch = [], []
    for i, d in enumerate(batch_data):
        curr_q, s_idx = d['current_q'], d['state_idx']

        if (curr_q, s_idx) not in pmm.pmm['delta']:
            mask = node_to_mask_t.get(curr_q)
            if mask is not None:
                tmp_dists = dists[i].clone()
                tmp_dists[~mask] = float('inf')
                s_idx = state_indices_array[torch.argmin(tmp_dists).item()]

        a_list = pmm.predict_list(curr_q, s_idx)
        action_lists_batch.append(a_list)
        cluster_centers_batch.append(cluster_centers_t[s_idx])

    cluster_centers_batch_t = torch.stack(cluster_centers_batch)
    return (action_lists_batch,
            cluster_centers_batch_t,
            concat_feat_t,
            torch.from_numpy(np.array([d['action_true'] for d in batch_data])).to(device))


def evaluate_model(model, all_eval_data, pmm, feat_array_t, state_indices_array, node_to_mask_t, cluster_centers_t, a_dim, device="cuda"):
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for i in range(0, len(all_eval_data), 256):
            batch_data = all_eval_data[i:i + 256]
            action_lists_t, cluster_center_t, concat_feat_t, action_true_t = \
                process_batch(batch_data, pmm, feat_array_t, state_indices_array, node_to_mask_t, cluster_centers_t, a_dim, device)
            log_prob = model.get_log_prob(action_lists_t, cluster_center_t, concat_feat_t, action_true_t)
            eval_losses.append(-log_prob.mean().item())
    model.train()
    return np.mean(eval_losses) if eval_losses else float('inf')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_residual_network(train_episodes, eval_episodes, cluster_centers, pmm, norm_stats,
                           epochs=300, lr=3e-4, device="cuda", model=None,
                           results_dir="results", checkpoint_dir="results/checkpoints"):
    first_step = next(ep[0] for ep in train_episodes if len(ep) > 0)
    a_dim = len(first_step['action'])
    feat_dim = len(norm_stats['rgb_mean']) + len(norm_stats['pos_mean'])
    print(f"Action dim: {a_dim}, Feature dim: {feat_dim}")

    print("Pre-processing and normalizing data...")
    all_train_data = collect_episode_data(train_episodes, pmm, norm_stats)
    all_eval_data = collect_episode_data(eval_episodes, pmm, norm_stats)

    print("Building GPU caches...")
    feature_indexes = build_feature_indexes_from_pmm(pmm, norm_stats)
    feat_array, state_indices_array = feature_indexes
    feat_array_t = torch.from_numpy(feat_array).float().to(device)

    node_to_mask_t = {}
    for q in pmm.pmm['Q']:
        valid_s = [s for (node, s) in pmm.pmm['delta'].keys() if node == q]
        if valid_s:
            mask = np.isin(state_indices_array, valid_s)
            node_to_mask_t[q] = torch.from_numpy(mask).to(device)

    cluster_centers_t = torch.from_numpy(cluster_centers).float().to(device)

    if model is None:
        model = ResidualMLP(feat_dim, a_dim, hidden=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eval_loss = float('inf')
    train_losses_history, eval_losses_history = [], []

    pbar = tqdm(range(epochs), desc="Training Residual Network")
    for epoch in pbar:
        np.random.shuffle(all_train_data)
        epoch_losses = []
        bs = 512
        for i in range(0, len(all_train_data), bs):
            batch_data = all_train_data[i:i + bs]
            action_lists_t, cluster_center_t, concat_feat_t, action_true_t = \
                process_batch(batch_data, pmm, feat_array_t, state_indices_array, node_to_mask_t, cluster_centers_t, a_dim, device)

            concat_feat_t += torch.randn_like(concat_feat_t) * 0.01

            log_prob = model.get_log_prob(action_lists_t, cluster_center_t, concat_feat_t, action_true_t)
            loss = -log_prob.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append(loss.item())

        eval_loss = evaluate_model(model, all_eval_data, pmm, feat_array_t, state_indices_array, node_to_mask_t, cluster_centers_t, a_dim, device)
        scheduler.step()

        avg_train_loss = np.mean(epoch_losses)
        train_losses_history.append(avg_train_loss)
        eval_losses_history.append(eval_loss)

        pbar.set_postfix({'train': f"{avg_train_loss:.4f}", 'eval': f"{eval_loss:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

        if optimizer.param_groups[0]["lr"] < 1e-7:
            print("Learning rate below 1e-7, stopping training.")
            break

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({'model': model.state_dict(), 'norm_stats': norm_stats},
                       os.path.join(checkpoint_dir, "residual_net.pt"))

    if train_losses_history:
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(train_losses_history) + 1)
        plt.plot(epochs_range, train_losses_history, 'o-', label='Train Loss', linewidth=2, markersize=4)
        plt.plot(epochs_range, eval_losses_history, 's-', label='Eval Loss', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12); plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14)
        plt.legend(fontsize=11); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved loss curves to {os.path.join(results_dir, 'loss_curves.png')}")

    return model, feature_indexes


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualize_actions(test_episodes, cluster_centers, pmm, model, feature_indexes, norm_stats,
                      device="cuda", n_vis=3, results_dir="results"):
    """Visualize ground-truth, PMM and residual-corrected actions."""
    model.eval()

    first_ep = next(ep for ep in test_episodes if len(ep) > 0)
    a_dim = len(first_ep[0]['action'])

    selected_indices = np.random.choice(len(test_episodes), min(n_vis, len(test_episodes)), replace=False)
    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(12, 4 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = [axes]

    for plot_idx, ep_idx in enumerate(selected_indices):
        episode = test_episodes[ep_idx]
        true_actions, pmm_actions, residual_actions = [], [], []

        h_seq = compute_episode_rnn_embeddings(episode, pmm)
        current_q = 0

        with torch.no_grad():
            for step_idx, step in enumerate(episode):
                rgb_n = normalize_feature(step['rgb_feature'], norm_stats['rgb_mean'], norm_stats['rgb_std'])
                pos_n = normalize_feature(step['pos_feature'], norm_stats['pos_mean'], norm_stats['pos_std'])
                concat_feat = np.concatenate([rgb_n, pos_n])

                state_idx = find_valid_state_idx(current_q, int(np.argmax(step['state'])),
                                                 concat_feat, pmm, feature_indexes)

                action_pmm_mean = pmm.predict(current_q, state_idx)
                a_list = [pmm.predict_list(current_q, state_idx)]

                current_q = update_pmm_node(pmm, current_q, state_idx, h_seq[step_idx])

                cc = cluster_centers[state_idx]
                cluster_center_t = torch.from_numpy(cc).float().to(device).unsqueeze(0)
                concat_feat_t = torch.from_numpy(concat_feat).float().to(device).unsqueeze(0)

                action_pred = model.get_action_mean(a_list, cluster_center_t, concat_feat_t).cpu().numpy()[0]

                true_actions.append(step['action'])
                pmm_actions.append(action_pmm_mean)
                residual_actions.append(action_pred)

        pca = PCA(n_components=1, random_state=42)
        true_array, pmm_array, residual_array = np.array(true_actions), np.array(pmm_actions), np.array(residual_actions)
        pca.fit(np.vstack([true_array, pmm_array, residual_array]))

        ax = axes[plot_idx]
        T = len(true_actions)
        ax.plot(range(T), pca.transform(true_array).flatten(), 'o-', label='True')
        ax.plot(range(T), pca.transform(pmm_array).flatten(), 's-', label='PMM', alpha=0.7)
        ax.plot(range(T), pca.transform(residual_array).flatten(), '^-', label='Residual', alpha=0.7)
        ax.set_title(f"Episode {ep_idx} Action Comparison")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "action_comparison.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", type=str, default="data/episodes_with_states.pkl")
    parser.add_argument("--rnn-weights", type=str, default="results/checkpoints/rnn_pretrained.pt")
    parser.add_argument("--trained-pt", type=str, default=None, help="Resume residual_net.pt if provided")
    parser.add_argument("--cos-tau", type=float, default=0.6, choices=[0.4, 0.6, 0.8])
    args = parser.parse_args()

    results_dir = "results"
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    episodes, cluster_centers = load_trajectories(args.input_pkl)

    train_episodes = episodes[:int(len(episodes) * 0.9)]
    test_episodes = episodes[int(len(episodes) * 0.9):]

    print("Computing normalization stats...")
    norm_stats = compute_normalization_stats(train_episodes)

    pmm_train_episodes = train_episodes[:10]
    pmm_trajs = [[{'state': s['state'], 'action': s['action']} for s in ep]
                 for ep in pmm_train_episodes]

    pmm = PMM(use_tqdm=True, max_inner_iters=1000)
    pmm.learn_pmm(pmm_trajs, rnn_weights_path=args.rnn_weights, cluster_centers=cluster_centers,
                   raw_episodes=pmm_train_episodes)
    pmm.save_dot(os.path.join(results_dir, "pmm_residual_training.dot"))
    pmm.save_pmm(os.path.join(results_dir, "pmm_residual_training.pkl"))
    print(f"PMM: {len(pmm.pmm['Q'])} nodes, {len(pmm.pmm['delta'])} edges")

    model, feature_indexes = train_residual_network(
        train_episodes, test_episodes, cluster_centers, pmm, norm_stats,
        device=device, model=None, results_dir=results_dir, checkpoint_dir=checkpoint_dir)

    visualize_actions(test_episodes, cluster_centers, pmm, model, feature_indexes, norm_stats,
                      device=device, results_dir=results_dir)
    print(f"Checkpoints saved to {checkpoint_dir}, other results saved to {results_dir}")


if __name__ == "__main__":
    main()
