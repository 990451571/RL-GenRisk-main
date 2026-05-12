import numpy as np


class PrioritizedReplayBuffer:
    def __init__(
        self,
        max_size,
        n_actions,
        alpha=0.2,
        beta_start=0.1,
        beta_frames=2000000,
        eps=1e-5,
    ):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.sample_step = 0

        self.memory_s = np.zeros((self.mem_size, self.n_actions, 3), dtype=np.float32)
        self.memory_a = np.zeros((self.mem_size, 1), dtype=np.int64)
        self.memory_r = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.memory_ai = np.zeros((self.mem_size, self.n_actions), dtype=np.float32)
        self.memory_sa = np.zeros((self.mem_size, self.n_actions), dtype=np.float32)

        self.priorities = np.zeros((self.mem_size,), dtype=np.float32)

    def store_transition(self, state, action, reward, action_index, sel_action):
        index = self.mem_cntr % self.mem_size

        self.memory_s[index, :] = state
        self.memory_a[index, :] = action
        self.memory_r[index, :] = reward
        self.memory_ai[index, :] = action_index
        self.memory_sa[index, :] = sel_action

        max_priority = self.priorities.max() if self.mem_cntr > 0 else 1.0
        if max_priority <= 0:
            max_priority = 1.0
        self.priorities[index] = max_priority

        self.mem_cntr += 1

    def beta_by_frame(self):
        progress = min(1.0, self.sample_step / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def sample_buffer(self, batch_size):
        current_size = min(self.mem_cntr, self.mem_size)

        if current_size == 0:
            raise ValueError("Replay buffer is empty.")

        priorities = self.priorities[:current_size]

        if priorities.sum() <= 0:
            probs = np.ones(current_size, dtype=np.float32) / current_size
        else:
            probs = priorities ** self.alpha
            probs_sum = probs.sum()

            if probs_sum <= 0 or np.isnan(probs_sum):
                probs = np.ones(current_size, dtype=np.float32) / current_size
            else:
                probs = probs / probs_sum

        replace = current_size < batch_size
        sample_index = np.random.choice(
            current_size,
            size=batch_size,
            replace=replace,
            p=probs,
        )

        self.sample_step += 1
        beta = self.beta_by_frame()

        weights = (current_size * probs[sample_index]) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32).reshape(-1, 1)

        batch_s = self.memory_s[sample_index, :]
        batch_a = self.memory_a[sample_index, :]
        batch_r = self.memory_r[sample_index, :]
        batch_ai = self.memory_ai[sample_index, :]
        batch_sa = self.memory_sa[sample_index, :]

        return batch_s, batch_a, batch_r, batch_ai, batch_sa, sample_index, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors).reshape(-1)

        for idx, err in zip(indices, td_errors):
            priority = np.clip(err + self.eps, 1e-5, 5.0)
            self.priorities[idx] = priority

    def clear(self):
        self.memory_s.fill(0)
        self.memory_a.fill(0)
        self.memory_r.fill(0)
        self.memory_ai.fill(0)
        self.memory_sa.fill(0)
        self.priorities.fill(0)
        self.mem_cntr = 0
        self.sample_step = 0

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)