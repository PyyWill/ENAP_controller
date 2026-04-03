import pickle
import numpy as np
import math
import random
import os
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import trange


# ---------------------------------------------------------------------------
# RNN Definitions
# ---------------------------------------------------------------------------
class TrainableRNN(nn.Module):
    def __init__(self, action_dim, state_dim, embed_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.state_embed = nn.Embedding(state_dim, embed_dim)
        self.rnn = nn.RNN(action_dim + embed_dim, h_dim, batch_first=True)

    def load_weights(self, state_dict):
        """Load weights from a training checkpoint (enc.* -> rnn.*)."""
        self.state_embed.weight.data = state_dict['state_embed.weight']
        self.rnn.weight_ih_l0.data = state_dict['enc.weight_ih_l0']
        self.rnn.weight_hh_l0.data = state_dict['enc.weight_hh_l0']
        self.rnn.bias_ih_l0.data = state_dict['enc.bias_ih_l0']
        self.rnn.bias_hh_l0.data = state_dict['enc.bias_hh_l0']

    def forward_internal(self, A, S):
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
            S = torch.from_numpy(S).long()
        if A.dim() == 2:
            A = A.unsqueeze(0)
            S = S.unsqueeze(0)
        s_emb = self.state_embed(S)
        x = torch.cat([A, s_emb], dim=-1)
        out, h_n = self.rnn(x)
        return out, h_n

    def encode(self, A, S):
        _, h_n = self.forward_internal(A, S)
        return h_n.squeeze(0).squeeze(0).detach().cpu().numpy()

    def forward_trajectory(self, A, S):
        out, _ = self.forward_internal(A, S)
        return out.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def forward_step(self, action, state_idx, h_prev=None):
        """Single-step RNN update.  Returns new hidden state as numpy (h_dim,)."""
        a = torch.from_numpy(np.asarray(action, dtype=np.float32)).reshape(1, 1, -1)
        s = torch.tensor([[state_idx]], dtype=torch.long)
        s_emb = self.state_embed(s)
        x = torch.cat([a, s_emb], dim=-1)          # (1, 1, in_dim)
        if h_prev is None:
            h0 = torch.zeros(1, 1, self.h_dim)
        else:
            h0 = torch.from_numpy(np.asarray(h_prev, dtype=np.float32)).reshape(1, 1, self.h_dim)
        _, h_n = self.rnn(x, h0)                    # h_n: (1, 1, h_dim)
        return h_n.squeeze().cpu().numpy()


class FrozenRNN:
    """Random-weight RNN fallback when no pretrained weights are available."""
    def __init__(self, in_dim, h_dim):
        self.Wx = np.random.randn(h_dim, in_dim) / math.sqrt(in_dim)
        self.Wh = np.random.randn(h_dim, h_dim) / math.sqrt(h_dim)
        self.b = np.zeros((h_dim,))
        self.h_dim = h_dim

    def encode(self, X, S=None):
        h = np.zeros((self.h_dim,))
        for x in X:
            h = np.tanh(self.Wx @ x + self.Wh @ h + self.b)
        return h

    def forward_trajectory(self, X, S=None):
        seq_len = X.shape[0]
        hs = np.zeros((seq_len, self.h_dim))
        h = np.zeros((self.h_dim,))
        for t in range(seq_len):
            h = np.tanh(self.Wx @ X[t] + self.Wh @ h + self.b)
            hs[t] = h
        return hs

    def forward_step(self, action, state_idx=None, h_prev=None):
        """Single-step update.  Returns new hidden state as numpy (h_dim,)."""
        x = np.asarray(action, dtype=np.float64).flatten()
        h = np.zeros(self.h_dim) if h_prev is None else np.asarray(h_prev, dtype=np.float64)
        return np.tanh(self.Wx @ x + self.Wh @ h + self.b)


# ---------------------------------------------------------------------------
# PMM Class
# ---------------------------------------------------------------------------
class PMM:
    def __init__(self, cos_tau_row=0.6, error_threshold=0.3, max_inner_iters=20,
                 stabil_required=2, use_observed_sigma=True, use_tqdm=True, seed=42):
        self.cos_tau_row = cos_tau_row
        self.error_threshold = error_threshold
        self.max_inner_iters = max_inner_iters
        self.stabil_required = stabil_required
        self.use_observed_sigma = use_observed_sigma
        self.use_tqdm = use_tqdm
        self.seed = seed

        self.episodes = []              # [{S: (T, s_dim), A: (T, a_dim)}, ...]
        self.sigma = []
        self.a_dim = None
        self.s_dim = None
        self.cluster_centers = None

        self.rnn = None
        self.rnn_hidden = 64
        self.rnn_embed_dim = 16

        # Transient embedding database (rebuilt on load)
        self.db_embeddings = None       # (N, H)
        self.db_embeddings_norm = None  # (N, H) row-normalised
        self.db_indices_map = []        # [(epi, t), ...]
        self._episode_offsets = []      # cumulative step offsets per episode

        # PMM topology  {Q, delta, reps, rep_embeddings}
        self.pmm = None
        self.S = []                     # L* prefix set (transient, only during learning)

        # Replay cache (rebuilt after learn / load)
        self._edge_cache = None         # {(q, x, q') -> [(epi, t), ...]}
        self._step_cache = None         # {(epi, t) -> (q, x, q')}
        self._qx_actions = None         # {(q, x) -> [action_array, ...]}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _init_rnn(self, rnn_weights_path=None):
        if self.episodes and self.a_dim is None and 'A' in self.episodes[0]:
            self.a_dim = self.episodes[0]['A'].shape[1]

        if not rnn_weights_path or not os.path.exists(rnn_weights_path):
            raise FileNotFoundError(
                f"RNN weights required but not found: {rnn_weights_path!r}")
        else:
            ckpt = torch.load(rnn_weights_path, map_location='cpu', weights_only=False)
            dims = ckpt['dims']
            self.a_dim = dims['a']
            self.s_dim = dims['s']
            self.rnn_embed_dim = dims['e']
            self.rnn_hidden = dims['h']
            self.rnn = TrainableRNN(self.a_dim, self.s_dim, self.rnn_embed_dim, self.rnn_hidden)
            self.rnn.load_weights(ckpt['model_state'])
            print(f"Loaded RNN weights from {rnn_weights_path}")

    def _build_sigma(self):
        if self.use_observed_sigma:
            self.sigma = sorted({int(np.argmax(ep['S'][t]))
                                 for ep in self.episodes for t in range(ep['S'].shape[0])})
        else:
            self.sigma = list(range(self.s_dim))

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _get_inputs_for_window(self, ep, t_end_inclusive):
        A = ep['A'][:t_end_inclusive + 1]
        S = np.argmax(ep['S'][:t_end_inclusive + 1], axis=1)
        return A, S

    def _precompute_embeddings(self):
        """Compute RNN hidden states for every timestep in every episode."""
        all_embeds = []
        self.db_indices_map = []
        self._episode_offsets = [0]

        for epi, ep in enumerate(self.episodes):
            T = ep['S'].shape[0]
            if T > 0:
                full_A, full_S = self._get_inputs_for_window(ep, T - 1)
                h_seq = self.rnn.forward_trajectory(full_A, full_S)
                for t in range(T):
                    all_embeds.append(h_seq[t])
                    self.db_indices_map.append((epi, t))
            self._episode_offsets.append(self._episode_offsets[-1] + T)

        self.db_embeddings = np.stack(all_embeds)
        norms = np.linalg.norm(self.db_embeddings, axis=1, keepdims=True) + 1e-12
        self.db_embeddings_norm = self.db_embeddings / norms

    def _get_step_embedding(self, epi, t):
        """Fast O(1) lookup into the precomputed embedding database."""
        if self.db_embeddings is None:
            return None
        idx = self._episode_offsets[epi] + t
        return self.db_embeddings[idx] if idx < len(self.db_embeddings) else None

    def _cosine_similarity(self, vec, matrix_norm):
        v_norm = vec / (np.linalg.norm(vec) + 1e-12)
        return matrix_norm @ v_norm

    def _cosine(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)

    def _get_representative_embedding(self, seq_prefix):
        """Canonical RNN embedding for a symbolic prefix sequence."""
        if len(seq_prefix) == 0:
            return np.zeros(self.rnn_hidden)

        seq_arr = np.array(seq_prefix)
        L = len(seq_arr)
        match = None

        for epi, ep in enumerate(self.episodes):
            Sidx = np.argmax(ep['S'], axis=1)
            for t in range(max(0, L - 1), len(Sidx)):
                if np.array_equal(Sidx[t - L + 1:t + 1], seq_arr):
                    match = (epi, t)
                    break
            if match:
                break

        if match is None:
            return None

        epi, t = match
        A, S = self._get_inputs_for_window(self.episodes[epi], t)
        z_seed = self.rnn.encode(A, S)

        sims = self._cosine_similarity(z_seed, self.db_embeddings_norm)
        mask = sims >= self.cos_tau_row
        if not np.any(mask):
            return z_seed
        return np.mean(self.db_embeddings[mask], axis=0)

    def _find_nearest_rep_index(self, z, rep_embeddings):
        best_idx, best_sim = 0, -2.0
        z_norm = z / (np.linalg.norm(z) + 1e-12)
        for i, z_rep in enumerate(rep_embeddings):
            sim = np.dot(z_norm, z_rep / (np.linalg.norm(z_rep) + 1e-12))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # L* components
    # ------------------------------------------------------------------
    def _ensure_closed(self, reps, rep_embeddings):
        for i, r_seq in enumerate(reps):
            z_r = rep_embeddings[i]

            if len(r_seq) == 0:
                indices = np.array([idx for idx, (_, t) in enumerate(self.db_indices_map) if t == 0])
            else:
                sims = self._cosine_similarity(z_r, self.db_embeddings_norm)
                mask = sims >= self.cos_tau_row
                if not np.any(mask):
                    continue
                indices = np.where(mask)[0]

            for idx in indices:
                epi, t = self.db_indices_map[idx]
                ep = self.episodes[epi]
                if t >= len(ep['S']) - 1:
                    continue

                x = int(np.argmax(ep['S'][t]))
                z_next = self.db_embeddings[idx]
                nearest_idx = self._find_nearest_rep_index(z_next, rep_embeddings)
                sim = self._cosine(z_next, rep_embeddings[nearest_idx])

                if sim < self.cos_tau_row:
                    new_seq = tuple(list(r_seq) + [x])
                    if new_seq not in reps:
                        new_seq_real = tuple(np.argmax(ep['S'][:t + 1], axis=1))
                        if not any(self._cosine(z_next, z_ex) >= self.cos_tau_row
                                   for z_ex in rep_embeddings):
                            self.S.append(new_seq_real)
                            return True
        return False

    def _build_pmm(self, S):
        """Build PMM hypothesis from prefix set S.

        Returns a dict with permanent topology (Q, delta, reps, rep_embeddings)
        plus temporary fields (_qx_actions, _edge_next_inputs) used only during
        L* iteration.
        """
        # 1. Representatives
        reps, rep_embeddings = [], []
        for s in S:
            z = self._get_representative_embedding(s)
            if z is None:
                continue
            if not any(self._cosine(z, z_ex) >= self.cos_tau_row for z_ex in rep_embeddings):
                reps.append(s)
                rep_embeddings.append(z)

        if not reps:
            reps = [tuple()]
            rep_embeddings = [np.zeros(self.rnn_hidden)]

        s_set = set(S)

        def _history_in_S(epi, t):
            ep = self.episodes[epi]
            history = tuple(np.argmax(ep['S'][:t], axis=1)) if t > 0 else ()
            return history in s_set

        z_init = np.zeros(self.rnn_hidden)
        q_init = self._find_nearest_rep_index(z_init, rep_embeddings)

        # 2. Scan embedding database
        edge_data = defaultdict(list)   # (q, x, q') -> [(action, next_x | None), ...]

        for idx in range(len(self.db_embeddings)):
            epi, t = self.db_indices_map[idx]
            if not _history_in_S(epi, t):
                continue

            q_prev = q_init if t == 0 else self._find_nearest_rep_index(
                self.db_embeddings[idx - 1], rep_embeddings)

            ep = self.episodes[epi]
            x = int(np.argmax(ep['S'][t]))
            action = ep['A'][t]
            q_next = self._find_nearest_rep_index(self.db_embeddings[idx], rep_embeddings)
            next_x = int(np.argmax(ep['S'][t + 1])) if t < len(ep['S']) - 1 else None

            edge_data[(q_prev, x, q_next)].append((action, next_x))

        # 3. Aggregate
        delta = {}
        qx_actions = defaultdict(list)
        edge_next_inputs = defaultdict(set)
        qx_dest_counts = defaultdict(lambda: defaultdict(int))

        for (q, x, q_next), samples in edge_data.items():
            qx_dest_counts[(q, x)][q_next] += len(samples)
            for action, next_x in samples:
                qx_actions[(q, x)].append(action)
                if next_x is not None:
                    edge_next_inputs[(q, x, q_next)].add(next_x)

        for (q, x), counts in qx_dest_counts.items():
            total = sum(counts.values())
            delta[(q, x)] = {nq: c / total for nq, c in counts.items()}

        return {
            'Q': list(range(len(reps))),
            'reps': reps,
            'delta': delta,
            'rep_embeddings': rep_embeddings,
            '_qx_actions': dict(qx_actions),
            '_edge_next_inputs': {k: sorted(v) for k, v in edge_next_inputs.items()},
        }

    def _equivalence_query(self, H):
        """NFA-style equivalence query: checks every episode against the hypothesis."""
        qx_actions = H['_qx_actions']
        rep_embeddings = H['rep_embeddings']

        for ep in self.episodes:
            T = ep['S'].shape[0]
            q_init = self._find_nearest_rep_index(np.zeros(self.rnn_hidden), rep_embeddings)
            possible_states = {q_init}

            for t in range(T):
                x = int(np.argmax(ep['S'][t]))
                true_action = ep['A'][t]
                next_possible = set()

                for q in possible_states:
                    actions = qx_actions.get((q, x))
                    if not actions:
                        continue
                    min_dist = np.min(np.linalg.norm(np.stack(actions) - true_action, axis=1))
                    if min_dist <= self.error_threshold:
                        for nq in H['delta'].get((q, x), {}):
                            next_possible.add(nq)

                possible_states = next_possible
                if not possible_states:
                    return self._extract_prefix(ep, t)
        return None

    def _extract_prefix(self, ep, t):
        return tuple(np.argmax(ep['S'][:t + 1], axis=1))

    def _merge_nodes(self, pmm):
        """Strict task-phase merge via Union-Find.

        Merges q_dest into q_src when:
          1. q_src has a self-loop for input x
          2. Both the self-loop and (q_src -> q_dest) have next_inputs == [x]
        """
        reps = pmm['reps']
        rep_embeddings = pmm['rep_embeddings']
        edge_next = pmm.get('_edge_next_inputs', {})
        n_nodes = len(reps)

        parent = list(range(n_nodes))

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj
                return True
            return False

        while True:
            changed = False
            for key, dist in pmm['delta'].items():
                q_src, x = key
                if q_src not in dist:
                    continue
                self_sig = tuple(sorted(edge_next.get((q_src, x, q_src), [])))
                if self_sig != (x,):
                    continue
                for q_dest in dist:
                    if q_dest == q_src:
                        continue
                    if tuple(sorted(edge_next.get((q_src, x, q_dest), []))) == self_sig:
                        if union(q_src, q_dest):
                            changed = True
            if not changed:
                break

        # Rebuild with merged nodes
        old_roots = sorted(set(find(i) for i in range(n_nodes)))
        root_to_new = {r: i for i, r in enumerate(old_roots)}
        new_n = len(old_roots)

        new_reps = [None] * new_n
        embed_groups = [[] for _ in range(new_n)]
        for i in range(n_nodes):
            r = root_to_new[find(i)]
            if new_reps[r] is None:
                new_reps[r] = reps[i]
            if i < len(rep_embeddings):
                embed_groups[r].append(rep_embeddings[i])

        new_rep_embeddings = [
            np.mean(g, axis=0) if g else np.zeros(self.rnn_hidden)
            for g in embed_groups
        ]

        new_delta = defaultdict(lambda: defaultdict(float))
        new_edge_next = defaultdict(set)

        for key, dist in pmm['delta'].items():
            q_old, x = key
            q_new = root_to_new[find(q_old)]
            for q_dest_old, prob in dist.items():
                nq_new = root_to_new[find(q_dest_old)]
                new_delta[(q_new, x)][nq_new] += prob
                old_eni = edge_next.get((q_old, x, q_dest_old), [])
                new_edge_next[(q_new, x, nq_new)].update(old_eni)

        final_delta = {}
        for key, dist in new_delta.items():
            total = sum(dist.values())
            final_delta[key] = {k: v / total for k, v in dist.items()}

        return {
            'Q': list(range(new_n)),
            'reps': new_reps,
            'delta': final_delta,
            'rep_embeddings': new_rep_embeddings,
            '_edge_next_inputs': {k: sorted(v) for k, v in new_edge_next.items()},
        }

    # ------------------------------------------------------------------
    # Main learning entry point
    # ------------------------------------------------------------------
    def learn_pmm(self, trajectory_batch, rnn_weights_path=None, cluster_centers=None,
                   raw_episodes=None):
        self._set_seed()
        self.cluster_centers = np.asarray(cluster_centers) if cluster_centers is not None else None

        self.episodes = []
        for ep in trajectory_batch:
            S = np.stack([t['state'] for t in ep], axis=0)
            A = np.stack([t['action'] for t in ep], axis=0)
            self.episodes.append({'S': S, 'A': A})

        self.raw_features = None
        if raw_episodes is not None:
            self.raw_features = [
                [{'rgb_feature': s['rgb_feature'], 'pos_feature': s['pos_feature']}
                 for s in ep]
                for ep in raw_episodes
            ]

        self.a_dim = len(trajectory_batch[0][0]['action'])
        self.s_dim = len(trajectory_batch[0][0]['state'])

        self._init_rnn(rnn_weights_path)
        self._build_sigma()
        self._precompute_embeddings()

        self.S = [tuple()]
        stabil = 0

        for it in trange(self.max_inner_iters, desc="L* Iteration", disable=not self.use_tqdm):
            # Closedness check
            while True:
                temp_reps, temp_embeds = [], []
                for s in self.S:
                    z = self._get_representative_embedding(s)
                    if z is not None and not any(
                        self._cosine(z, z_ex) >= self.cos_tau_row for z_ex in temp_embeds
                    ):
                        temp_reps.append(s)
                        temp_embeds.append(z)
                if not self._ensure_closed(temp_reps, temp_embeds):
                    break

            H = self._build_pmm(self.S)
            ce = self._equivalence_query(H)

            if ce is None:
                stabil += 1
                print(f"----> Stabilized at iteration {it + 1}")
                if stabil >= self.stabil_required:
                    H = self._merge_nodes(H)
                    self.pmm = {
                        'Q': H['Q'], 'delta': H['delta'],
                        'reps': H['reps'], 'rep_embeddings': H['rep_embeddings'],
                    }
                    self._rebuild_cache()
                    self._prune_after_replay()
                    print(f"----> Converged at iteration {it + 1}")
                    return self.pmm
            else:
                for k in range(1, len(ce) + 1):
                    p = ce[:k]
                    if p not in self.S:
                        self.S.append(p)
                stabil = 0

        # Max iterations reached
        H = self._build_pmm(self.S)
        H = self._merge_nodes(H)
        self.pmm = {
            'Q': H['Q'], 'delta': H['delta'],
            'reps': H['reps'], 'rep_embeddings': H['rep_embeddings'],
        }
        self._rebuild_cache()
        self._prune_after_replay()
        return self.pmm

    # ------------------------------------------------------------------
    # Replay (embedding-based disambiguation)
    # ------------------------------------------------------------------
    def replay_assign(self):
        """Walk each episode through delta, assigning (epi, t) to edges.

        Disambiguation strategy:
          - 1 outgoing edge  -> follow it
          - multiple edges   -> pick destination whose rep_embedding is most
                                similar to the RNN embedding at (epi, t)
          - no matching edge -> self-loop
        """
        if not self.episodes or self.pmm is None:
            return {}, {}

        delta = self.pmm['delta']
        rep_embeddings = self.pmm.get('rep_embeddings', [])

        edge_samples = defaultdict(list)
        step_to_edge = {}

        for epi, ep in enumerate(self.episodes):
            T = ep['S'].shape[0]
            q = 0
            for t in range(T):
                x = int(np.argmax(ep['S'][t]))
                dist = delta.get((q, x))

                if dist is None:
                    q_next = q
                elif len(dist) == 1:
                    q_next = next(iter(dist))
                else:
                    z_t = self._get_step_embedding(epi, t)
                    best_q, best_sim = None, -2.0
                    if z_t is not None and rep_embeddings:
                        z_t_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
                        for q_cand in dist:
                            if q_cand < len(rep_embeddings):
                                sim = np.dot(z_t_norm,
                                             rep_embeddings[q_cand] / (np.linalg.norm(rep_embeddings[q_cand]) + 1e-12))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_q = q_cand
                    if best_q is None:
                        raise RuntimeError("PMM multi-edge disambiguation failed")
                    q_next = best_q

                edge_samples[(q, x, q_next)].append((epi, t))
                step_to_edge[(epi, t)] = (q, x, q_next)
                q = q_next

        return dict(edge_samples), step_to_edge

    def _rebuild_cache(self):
        """Run replay and populate action lookup caches."""
        self._edge_cache, self._step_cache = self.replay_assign()
        self._qx_actions = defaultdict(list)
        for (q, x, _), indices in self._edge_cache.items():
            for epi, t in indices:
                self._qx_actions[(q, x)].append(self.episodes[epi]['A'][t])
        self._qx_actions = dict(self._qx_actions)

    def _prune_after_replay(self):
        """Remove empty edges and unreachable nodes, remap indices, recompute probabilities."""
        old_delta = self.pmm['delta']
        edge_cache = self._edge_cache or {}

        # 1. Rebuild delta from edge_cache sample counts (drops edges with 0 samples)
        qx_counts = defaultdict(lambda: defaultdict(int))
        for (q, x, q_next), indices in edge_cache.items():
            if indices:
                qx_counts[(q, x)][q_next] += len(indices)

        new_delta = {}
        for (q, x), counts in qx_counts.items():
            total = sum(counts.values())
            new_delta[(q, x)] = {nq: c / total for nq, c in counts.items()}

        # 2. BFS from q=0 to find reachable nodes
        reachable = set()
        frontier = [0]
        while frontier:
            node = frontier.pop(0)
            if node in reachable:
                continue
            reachable.add(node)
            for (q, x), dist in new_delta.items():
                if q == node:
                    for q_next in dist:
                        if q_next not in reachable:
                            frontier.append(q_next)

        # 3. Remap to contiguous indices
        old_sorted = sorted(reachable)
        old_to_new = {old: new for new, old in enumerate(old_sorted)}

        remapped_delta = {}
        for (q, x), dist in new_delta.items():
            if q not in reachable:
                continue
            new_dist = {}
            for q_next, prob in dist.items():
                if q_next in reachable:
                    new_dist[old_to_new[q_next]] = prob
            if new_dist:
                total = sum(new_dist.values())
                remapped_delta[(old_to_new[q], x)] = {nq: p / total for nq, p in new_dist.items()}

        old_reps = self.pmm['reps']
        old_rep_emb = self.pmm['rep_embeddings']
        new_reps = [old_reps[i] for i in old_sorted]
        new_rep_emb = [old_rep_emb[i] for i in old_sorted if i < len(old_rep_emb)]

        n_removed_nodes = len(self.pmm['Q']) - len(reachable)
        old_edge_count = sum(len(d) for d in old_delta.values())
        new_edge_count = sum(len(d) for d in remapped_delta.values())

        if n_removed_nodes > 0 or old_edge_count != new_edge_count:
            print(f"Prune: removed {n_removed_nodes} node(s), "
                  f"{old_edge_count - new_edge_count} edge(s) "
                  f"-> {len(reachable)} nodes, {new_edge_count} edges")

        self.pmm = {
            'Q': list(range(len(reachable))),
            'reps': new_reps,
            'delta': remapped_delta,
            'rep_embeddings': new_rep_emb,
        }
        self._rebuild_cache()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, q, x):
        """Return mean action for (q, x) from replay cache."""
        if self.pmm is None:
            raise ValueError("PMM not trained. Call learn_pmm() first.")
        if self._qx_actions is None:
            raise ValueError("No replay cache.")
        actions = self._qx_actions.get((q, x))
        if not actions:
            raise KeyError(f"No actions for (q={q}, x={x}) in replay cache. "
                           f"Available keys with q={q}: {[k for k in self._qx_actions if k[0]==q]}")
        return np.mean(actions, axis=0)

    def predict_list(self, q, x):
        """Return list of all actions for (q, x) from replay cache."""
        if self.pmm is None:
            raise ValueError("PMM not trained. Call learn_pmm() first.")
        if self._qx_actions is None:
            raise ValueError("No replay cache.")
        actions = self._qx_actions.get((q, x))
        if not actions:
            raise KeyError(f"No actions for (q={q}, x={x}) in replay cache. "
                           f"Available keys with q={q}: {[k for k in self._qx_actions if k[0]==q]}")
        return actions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pmm(self, path):
        """Save PMM as a single pkl: topology + rep_embeddings + RNN weights + episodes."""
        if self.pmm is None:
            raise ValueError("PMM not trained; nothing to save.")

        rnn_state, rnn_type = None, 'frozen'
        if isinstance(self.rnn, TrainableRNN):
            rnn_state = {k: v.detach().cpu().numpy() for k, v in self.rnn.state_dict().items()}
            rnn_type = 'trainable'
        elif isinstance(self.rnn, FrozenRNN):
            rnn_state = {'Wx': self.rnn.Wx, 'Wh': self.rnn.Wh, 'b': self.rnn.b}

        payload = {
            'pmm': self.pmm,            # {Q, delta, reps, rep_embeddings}
            'sigma': self.sigma,
            'a_dim': self.a_dim,
            's_dim': self.s_dim,
            'rnn_hidden': self.rnn_hidden,
            'rnn_embed_dim': self.rnn_embed_dim,
            'rnn_type': rnn_type,
            'rnn_state': rnn_state,
            'cos_tau_row': self.cos_tau_row,
            'cluster_centers': self.cluster_centers,
            'episodes': self.episodes,   # [{S, A}, ...]
            'raw_features': getattr(self, 'raw_features', None),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved PMM to {path}")

    def load_pmm(self, path):
        """Load PMM pkl, reconstruct RNN, recompute embeddings, rebuild replay cache."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        required_keys = ['pmm', 'sigma', 'a_dim', 's_dim', 'rnn_hidden',
                         'rnn_embed_dim', 'rnn_type', 'rnn_state',
                         'cos_tau_row', 'cluster_centers', 'episodes']
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(f"PMM pkl missing required keys: {missing}")

        pmm_data = payload['pmm']
        for k in ('Q', 'delta', 'reps', 'rep_embeddings'):
            assert k in pmm_data, f"pmm_data missing key '{k}'"
        self.pmm = {
            'Q': pmm_data['Q'],
            'delta': pmm_data['delta'],
            'reps': pmm_data['reps'],
            'rep_embeddings': pmm_data['rep_embeddings'],
        }
        self.sigma = payload['sigma']
        self.a_dim = payload['a_dim']
        self.s_dim = payload['s_dim']
        self.rnn_hidden = payload['rnn_hidden']
        self.rnn_embed_dim = payload['rnn_embed_dim']
        self.cos_tau_row = payload['cos_tau_row']
        self.cluster_centers = payload['cluster_centers']
        self.episodes = payload['episodes']
        self.raw_features = payload.get('raw_features', None)
        assert self.episodes, "PMM pkl contains no episodes"
        assert self.cluster_centers is not None, "PMM pkl missing cluster_centers"

        rnn_type = payload['rnn_type']
        rnn_state = payload['rnn_state']
        assert rnn_state is not None, "PMM pkl missing rnn_state"
        if rnn_type == 'trainable':
            self.rnn = TrainableRNN(self.a_dim, self.s_dim, self.rnn_embed_dim, self.rnn_hidden)
            self.rnn.load_state_dict({k: torch.from_numpy(v) for k, v in rnn_state.items()})
        elif rnn_type == 'frozen':
            self.rnn = FrozenRNN(self.a_dim + self.rnn_embed_dim, self.rnn_hidden)
            self.rnn.Wx = rnn_state['Wx']
            self.rnn.Wh = rnn_state['Wh']
            self.rnn.b = rnn_state['b']
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type!r}")

        self._precompute_embeddings()
        self._rebuild_cache()

        print(f"Loaded PMM from {path}")
        return self.pmm

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def save_dot(self, filename="pmm.dot"):
        """Export the automaton to Graphviz DOT format.

        Edge labels, avg actions and next_inputs are all derived on-the-fly
        from the replay cache + raw episodes.
        """
        if self.pmm is None:
            return

        Q = self.pmm['Q']
        reps = self.pmm['reps']
        DELTA = self.pmm['delta']
        state_labels = [f"c{i}" for i in range(self.s_dim)]

        lines = ["digraph PMM {", "  rankdir=LR;", "  node [shape=circle, fontsize=11];"]

        for q in Q:
            rep_str = "-".join(str(s) for s in reps[q]) if reps[q] else "ε"
            lines.append(f'  q{q} [label="q{q}\\nrep: {rep_str}"];')

        for q in Q:
            for x in self.sigma:
                if (q, x) not in DELTA:
                    continue

                for next_q, prob in DELTA[(q, x)].items():
                    edge_key = (q, x, next_q)
                    indices = self._edge_cache.get(edge_key, []) if self._edge_cache else []

                    # Per-edge avg action (from this specific edge, not shared across all outgoing edges)
                    if indices and self.episodes:
                        edge_actions = [self.episodes[epi]['A'][t] for epi, t in indices]
                        avg_act = np.mean(edge_actions, axis=0)
                        act_str = "[" + ", ".join(f"{v:.2f}" for v in avg_act[:2]) + "]"
                    else:
                        act_str = "[]"

                    next_inputs = set()
                    if indices and self.episodes:
                        for epi, t in indices:
                            ep = self.episodes[epi]
                            if t + 1 < ep['S'].shape[0]:
                                next_inputs.add(int(np.argmax(ep['S'][t + 1])))

                    x_label = state_labels[x] if x < len(state_labels) else str(x)
                    if next_inputs:
                        ni_str = ",".join(
                            state_labels[xi] if xi < len(state_labels) else str(xi)
                            for xi in sorted(next_inputs)
                        )
                        lbl = f"{x_label} / {act_str} p={prob:.2f} next: [{ni_str}]"
                    else:
                        lbl = f"{x_label} / {act_str} p={prob:.2f} next: []"

                    lines.append(f'  q{q} -> q{next_q} [label="{lbl}"];')

        lines.append("}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved DOT to {filename}")
