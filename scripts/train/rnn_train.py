import pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse, os


CHECKPOINT_DIR = os.path.join("results", "checkpoints")


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


class Pretrain(nn.Module):
    def __init__(self, action_dim, state_dim, embed_dim, h_dim):
        super().__init__()
        self.state_embed = nn.Embedding(state_dim, embed_dim)
        self.enc = nn.RNN(action_dim + embed_dim, h_dim, batch_first=True)
        self.act_head = nn.Linear(h_dim, action_dim)
        self.cls_head = nn.Linear(h_dim, state_dim)

    def forward_sequence(self, actions, state_indices):
        s_emb = self.state_embed(state_indices)
        x = torch.cat([actions, s_emb], dim=-1)
        out, _ = self.enc(x)
        return out

    def h(self, actions, state_indices):
        out = self.forward_sequence(actions.unsqueeze(0), state_indices.unsqueeze(0))
        return out.squeeze(0)[-1]


class PrioritizedReplayBuffer:
    def __init__(self, trajs, alpha=0.6):
        self.trajs = trajs
        self.alpha = alpha
        self.epsilon = 1e-6
        self.sample_indices = []
        for i, tr in enumerate(self.trajs):
            if len(tr["action"]) >= 2:
                for t in range(len(tr["action"]) - 1):
                    self.sample_indices.append((i, t))
        self.size = len(self.sample_indices)
        self.priorities = np.ones(self.size)

    def sample(self, batch_size, beta=0.4):
        probs = self.priorities ** self.alpha
        probs /= probs.sum()
        sampled_idxs = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[sampled_idxs]) ** (-beta)
        weights /= weights.max()
        return sampled_idxs, torch.from_numpy(weights).float()

    def update_priorities(self, indices, losses):
        self.priorities[indices] = np.abs(losses) + self.epsilon


def build_batch_from_indices(trajs, sample_indices, sampled_idxs, dev):
    hist_actions, hist_states = [], []
    target_a_tp1, target_s_tp1 = [], []
    s_t_labels, s_tp1_labels = [], []

    for idx in sampled_idxs:
        traj_idx, t = sample_indices[idx]
        tr = trajs[traj_idx]
        A, S = tr["action"], tr["state"]
        S_labels = np.argmax(S, axis=1)

        hist_actions.append(torch.tensor(A[:t+2], dtype=torch.float32, device=dev))
        hist_states.append(torch.tensor(S_labels[:t+2], dtype=torch.long, device=dev))
        target_a_tp1.append(torch.tensor(A[t+1], dtype=torch.float32, device=dev))
        target_s_tp1.append(torch.tensor(S_labels[t+1], dtype=torch.long, device=dev))
        s_t_labels.append(S_labels[t])
        s_tp1_labels.append(S_labels[t+1])

    padded_actions = pad_sequence(hist_actions, batch_first=True)
    padded_states = pad_sequence(hist_states, batch_first=True)
    return (padded_actions, padded_states,
            torch.stack(target_a_tp1), torch.stack(target_s_tp1),
            torch.tensor(s_t_labels, device=dev), torch.tensor(s_tp1_labels, device=dev))


def phase_aware_contrastive_loss(h_t, h_tp1, s_t, s_tp1, margin=0.5):
    """
    Self-Loop (s_t == s_tp1): Pull h_t and h_tp1 together (cosine dist -> 0).
    Transition (s_t != s_tp1): Push h_t and h_tp1 apart (cosine dist -> margin).
    """
    cosine_sim = (F.normalize(h_t, dim=1) * F.normalize(h_tp1, dim=1)).sum(dim=1)
    is_same = (s_t == s_tp1).float()
    dist = 1.0 - cosine_sim
    loss = is_same * dist + (1 - is_same) * torch.relu(margin - dist)
    return loss.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default="data/episodes_with_states.pkl")
    parser.add_argument("--loss-mode", type=str, choices=["all", "no-pred", "no-contrastive"], default="all")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42; np.random.default_rng(seed); torch.manual_seed(seed)

    bs = 512; epochs = 32; steps = 128; lr = 3e-4
    state_embed_dim = 16; h_dim = 64
    alpha = 0.6; beta_start = 0.4
    beta_frames = epochs * steps
    beta_by_frame = lambda f: min(1.0, beta_start + f * (1.0 - beta_start) / beta_frames)

    lam_pred_a, lam_pred_s, lam_contrast = 1.0, 1.0, 1.0
    if args.loss_mode == "no-pred":
        lam_pred_a, lam_pred_s = 0.0, 0.0
    elif args.loss_mode == "no-contrastive":
        lam_contrast = 0.0

    payload = load_pickle(args.pkl)
    trajs = [{"action": np.array([s["action"] for s in ep]),
              "state": np.array([s["state"] for s in ep])}
             for ep in payload["episodes"]]

    A_dim = trajs[0]["action"].shape[1]
    S_dim = trajs[0]["state"].shape[1]

    replay_buffer = PrioritizedReplayBuffer(trajs, alpha)
    model = Pretrain(A_dim, S_dim, state_embed_dim, h_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

    global_step = 0
    for ep in trange(epochs, desc="epochs"):
        avg = 0.0
        for step_idx in range(steps):
            beta = beta_by_frame(global_step)
            sampled_idxs, is_weights = replay_buffer.sample(bs, beta)
            is_weights = is_weights.to(device)

            actions, states, target_a, target_s, s_t, s_tp1 = build_batch_from_indices(
                trajs, replay_buffer.sample_indices, sampled_idxs, device
            )

            rnn_out = model.forward_sequence(actions, states)
            lengths = torch.tensor(
                [len(trajs[replay_buffer.sample_indices[idx][0]]["action"][:replay_buffer.sample_indices[idx][1]+2])
                 for idx in sampled_idxs],
                device=device,
            )
            batch_indices = torch.arange(bs, device=device)
            h_tp1 = rnn_out[batch_indices, lengths - 1]
            h_t = rnn_out[batch_indices, lengths - 2]

            loss_act_vec = F.mse_loss(model.act_head(h_t), target_a, reduction='none').mean(dim=1)
            loss_act = (loss_act_vec * is_weights).mean()
            loss_cls_vec = F.cross_entropy(model.cls_head(h_t), target_s, reduction='none')
            loss_cls = (loss_cls_vec * is_weights).mean()
            loss_cont = phase_aware_contrastive_loss(h_t, h_tp1, s_t, s_tp1)

            loss = lam_pred_a * loss_act + lam_pred_s * loss_cls + lam_contrast * loss_cont
            opt.zero_grad(); loss.backward(); opt.step()

            replay_buffer.update_priorities(sampled_idxs, (loss_act_vec + loss_cls_vec).detach().cpu().numpy())
            avg += loss.item()
            global_step += 1

        scheduler.step()
        # print(f"epoch {ep+1}: loss {avg/steps:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")

    out_path = os.path.join(CHECKPOINT_DIR, "rnn_pretrained.pt")
    torch.save({
        'model_state': model.state_dict(),
        'dims': {'a': A_dim, 's': S_dim, 'e': state_embed_dim, 'h': h_dim}
    }, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
